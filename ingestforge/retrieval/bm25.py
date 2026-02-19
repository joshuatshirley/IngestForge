"""
BM25 lexical search retriever.

Fast keyword-based retrieval using BM25 scoring.
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.config import Config
from ingestforge.storage.base import ChunkRepository, SearchResult


@dataclass
class BM25Params:
    """BM25 algorithm parameters."""

    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Length normalization
    delta: float = 0.5  # BM25+ delta


class BM25Retriever:
    """
    BM25 lexical search retriever.

    Implements BM25+ scoring for keyword-based retrieval.
    """

    def __init__(
        self,
        config: Config,
        storage: Optional[ChunkRepository] = None,
        params: Optional[BM25Params] = None,
    ):
        """
        Initialize BM25 retriever.

        Args:
            config: IngestForge configuration
            storage: Optional storage backend
            params: Optional BM25 parameters
        """
        self.config = config
        self.storage = storage
        self.params = params or BM25Params()

        # Index structures
        self._documents: Dict[str, str] = {}  # chunk_id -> content
        self._doc_lengths: Dict[str, int] = {}  # chunk_id -> length
        self._term_freqs: Dict[str, Dict[str, int]] = {}  # term -> {chunk_id: freq}
        self._doc_freqs: Dict[str, int] = {}  # term -> doc count
        self._avg_doc_length: float = 0.0
        self._loaded: bool = False

        # Stop words
        self._stop_words: Set[str] = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "and",
            "or",
            "but",
            "if",
            "than",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
            "we",
            "us",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "not",
        }

    @property
    def loaded(self) -> bool:
        """Check if index is loaded."""
        return self._loaded

    def index_chunks(self, chunks: List[ChunkRecord]) -> None:
        """
        Index chunks for search.

        Args:
            chunks: Chunks to index
        """
        for chunk in chunks:
            self._index_document(chunk.chunk_id, chunk.content)

        # Calculate average document length
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths.values()) / len(
                self._doc_lengths
            )

        self._loaded = True

    def _index_document(self, doc_id: str, content: str) -> None:
        """Index a single document."""
        tokens = self._tokenize(content)
        self._documents[doc_id] = content
        self._doc_lengths[doc_id] = len(tokens)

        # Count term frequencies
        term_counts: Dict[str, int] = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1

        # Update index
        for term, freq in term_counts.items():
            if term not in self._term_freqs:
                self._term_freqs[term] = {}
                self._doc_freqs[term] = 0

            self._term_freqs[term][doc_id] = freq
            self._doc_freqs[term] += 1

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return [t for t in tokens if t not in self._stop_words and len(t) > 1]

    def _calculate_bm25(
        self,
        query_terms: List[str],
        doc_id: str,
    ) -> float:
        """Calculate BM25+ score for a document."""
        n_docs = len(self._documents)
        doc_length = self._doc_lengths.get(doc_id, 1)

        score = 0.0
        for term in query_terms:
            if term not in self._term_freqs:
                continue

            if doc_id not in self._term_freqs[term]:
                continue

            # Term frequency in document
            tf = self._term_freqs[term][doc_id]

            # Document frequency
            df = self._doc_freqs.get(term, 0)

            # IDF component
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

            # BM25 TF component with length normalization
            length_norm = (
                1 - self.params.b + self.params.b * (doc_length / self._avg_doc_length)
            )
            tf_component = (tf * (self.params.k1 + 1)) / (
                tf + self.params.k1 * length_norm
            )

            # BM25+ adds delta to avoid zero scores
            score += idf * (tf_component + self.params.delta)

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        library_filter: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search using BM25.

        Rule #1: Reduced nesting from 4 → 2 levels via extraction
        Rule #4: Reduced from 71 → 34 lines

        Args:
            query: Search query
            top_k: Number of results
            library_filter: If provided, only return chunks from this library

        Returns:
            List of SearchResult
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        candidates = self._find_candidates(query_terms)
        if not candidates:
            return []

        scores = self._score_candidates(query_terms, candidates)
        scores.sort(key=lambda x: -x[1])
        return self._build_search_results(scores, top_k, library_filter)

    def _find_candidates(self, query_terms: List[str]) -> Set[str]:
        """Rule #1: Extracted candidate finding (<60 lines, 2 nesting levels)."""
        candidates: Set[str] = set()
        for term in query_terms:
            if term in self._term_freqs:
                candidates.update(self._term_freqs[term].keys())
        return candidates

    def _score_candidates(
        self, query_terms: List[str], candidates: Set[str]
    ) -> List[tuple]:
        """Rule #1: Extracted scoring logic (<60 lines, 2 nesting levels)."""
        scores = []
        for doc_id in candidates:
            score = self._calculate_bm25(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        return scores

    def _build_search_results(
        self, scores: List[tuple], top_k: int, library_filter: Optional[str]
    ) -> List[SearchResult]:
        """
        Build search results from scored documents.

        Rule #1: Reduced nesting from 4 → 2 levels
        Rule #4: Function <60 lines
        """
        results = []
        for doc_id, score in scores:
            if len(results) >= top_k:
                break

            result = self._create_search_result(doc_id, score, library_filter)
            if result:
                results.append(result)

        return results

    def _create_search_result(
        self, doc_id: str, score: float, library_filter: Optional[str]
    ) -> Optional[SearchResult]:
        """
        Create search result for a document.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines

        Returns:
            SearchResult or None if filtered out
        """
        # Try storage-based result first
        if self.storage:
            chunk = self.storage.get_chunk(doc_id)
            if chunk:
                if library_filter and chunk.library != library_filter:
                    return None
                return SearchResult.from_chunk(chunk, score)

        # Fallback: minimal result
        content = self._documents.get(doc_id, "")
        return SearchResult(
            chunk_id=doc_id,
            content=content,
            score=score,
            document_id="",
            section_title="",
            chunk_type="content",
            source_file="",
            word_count=len(content.split()),
        )
