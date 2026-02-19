"""Query rewriting for improved retrieval.

Rewrites queries to improve retrieval quality by expanding, reformulating,
and clarifying the original query."""

from __future__ import annotations

from typing import List, Dict, Any


class QueryRewriter:
    """Rewrite queries for better retrieval."""

    def __init__(self) -> None:
        """Initialize query rewriter."""
        self.stop_words = self._load_stop_words()

    def rewrite(self, query: str, strategy: str = "expand") -> Dict[str, Any]:
        """Rewrite query using specified strategy.

        Args:
            query: Original query
            strategy: Rewriting strategy ('expand', 'simplify', 'clarify')

        Returns:
            Dictionary with original and rewritten queries
        """
        if strategy == "expand":
            rewritten = self._expand_query(query)
        elif strategy == "simplify":
            rewritten = self._simplify_query(query)
        elif strategy == "clarify":
            rewritten = self._clarify_query(query)
        else:
            rewritten = [query]

        return {
            "original": query,
            "strategy": strategy,
            "rewritten": rewritten,
            "count": len(rewritten),
        }

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with variations.

        Args:
            query: Original query

        Returns:
            List of expanded queries
        """
        queries = [query]

        # Add question variations
        if "?" not in query:
            queries.append(f"What is {query}?")
            queries.append(f"Explain {query}")
            queries.append(f"How does {query} work?")

        # Add context variations
        queries.append(f"{query} overview")
        queries.append(f"{query} examples")
        queries.append(f"{query} definition")

        return queries[:5]  # Limit to 5 variations

    def _simplify_query(self, query: str) -> List[str]:
        """Simplify query by removing noise words.

        Args:
            query: Original query

        Returns:
            List with simplified query
        """
        words = query.lower().split()

        # Remove stop words
        key_words = [w for w in words if w not in self.stop_words]

        if not key_words:
            return [query]

        simplified = " ".join(key_words)
        return [query, simplified] if simplified != query.lower() else [query]

    def _clarify_query(self, query: str) -> List[str]:
        """Clarify query by making it more specific.

        Args:
            query: Original query

        Returns:
            List of clarified queries
        """
        queries = [query]

        # Add specificity
        words = query.lower().split()

        if len(words) < 3:
            # Short query - add context
            queries.append(f"{query} detailed explanation")
            queries.append(f"{query} key concepts")

        if any(w in words for w in ["what", "how", "why", "when", "where"]):
            # Already a question - add depth
            queries.append(f"{query} in detail")

        return queries[:3]

    def _load_stop_words(self) -> set[str]:
        """Load stop words.

        Returns:
            Set of stop words
        """
        return {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "must",
            "can",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "from",
            "as",
            "into",
        }


def rewrite_query(query: str, strategy: str = "expand") -> Dict[str, Any]:
    """Rewrite query for better retrieval.

    Args:
        query: Original query
        strategy: Rewriting strategy

    Returns:
        Dictionary with rewritten queries
    """
    rewriter = QueryRewriter()
    return rewriter.rewrite(query, strategy)


def multi_strategy_rewrite(query: str) -> Dict[str, List[str]]:
    """Rewrite query using multiple strategies.

    Args:
        query: Original query

    Returns:
        Dictionary mapping strategies to rewritten queries
    """
    rewriter = QueryRewriter()

    return {
        "original": [query],
        "expand": rewriter._expand_query(query),
        "simplify": rewriter._simplify_query(query),
        "clarify": rewriter._clarify_query(query),
    }
