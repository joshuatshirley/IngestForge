"""Base class for content analysis commands.

Provides common functionality for analyzing content patterns and usage.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from collections import Counter

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class AnalyzeCommand(IngestForgeCommand):
    """Base class for content analysis commands."""

    def get_llm_client(self, ctx: dict) -> Optional[Any]:
        """Get LLM client for analysis.

        Args:
            ctx: Context dictionary with config

        Returns:
            LLM client instance or None
        """
        try:
            from ingestforge.llm.factory import get_best_available_client

            client = get_best_available_client(ctx["config"])

            if client is None:
                self.print_warning(
                    "No LLM available for enhanced analysis.\n"
                    "Basic analysis will be performed."
                )

            return client

        except Exception as e:
            self.print_warning(f"Failed to load LLM client: {e}")
            return None

    def get_all_chunks_from_storage(self, storage: Any) -> list[Any]:
        """Retrieve all chunks from storage.

        Args:
            storage: ChunkRepository instance

        Returns:
            List of all chunks
        """
        return ProgressManager.run_with_spinner(
            lambda: self._retrieve_all_chunks(storage),
            "Retrieving chunks from storage...",
            "Chunks retrieved",
        )

    def _retrieve_all_chunks(self, storage: Any) -> list[Any]:
        """Retrieve all chunks (internal helper).

        Args:
            storage: ChunkRepository instance

        Returns:
            List of chunks
        """
        # Different storage backends have different APIs
        if hasattr(storage, "get_all_chunks"):
            return storage.get_all_chunks()
        elif hasattr(storage, "list_all"):
            return storage.list_all()
        else:
            # Fallback: search with empty query
            return storage.search("", k=10000)

    def extract_chunk_text(self, chunk: Any) -> str:
        """Extract text from chunk.

        Args:
            chunk: Chunk object or dict

        Returns:
            Chunk text content
        """
        # Try dict keys first
        if isinstance(chunk, dict):
            # Try content first (ChromaDB), then text
            return chunk.get("content", "") or chunk.get("text", "")

        # Try object attributes - content first (ChromaDB), then text
        for attr in ("content", "text"):
            if hasattr(chunk, attr):
                value = getattr(chunk, attr, None)
                if value:
                    return value

        # Last resort: try to get meaningful string representation
        # Avoid just str(chunk) which dumps metadata
        return ""

    def extract_chunk_metadata(self, chunk: Any) -> Dict[str, Any]:
        """Extract metadata from chunk.

        Args:
            chunk: Chunk object or dict

        Returns:
            Metadata dictionary
        """
        if isinstance(chunk, dict):
            # Try nested metadata first, then return whole dict
            return chunk.get("metadata", chunk)

        # For SearchResult or ChunkRecord objects, build metadata from attributes
        metadata = {}

        # Common metadata fields to extract from objects
        metadata_fields = [
            "source_file",
            "source",
            "document_id",
            "section_title",
            "chunk_type",
            "word_count",
            "library",
            "page_start",
            "page_end",
        ]

        for field in metadata_fields:
            if hasattr(chunk, field):
                value = getattr(chunk, field, None)
                if value is not None:
                    metadata[field] = value

        # Also include any explicit metadata dict
        if hasattr(chunk, "metadata") and chunk.metadata:
            if isinstance(chunk.metadata, dict):
                metadata.update(chunk.metadata)

        return metadata

    def extract_keywords_simple(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords using simple frequency analysis.

        Args:
            text: Text to analyze
            top_n: Number of top keywords to return

        Returns:
            List of keywords
        """
        # Simple keyword extraction (stopwords filtered)
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
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
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "their",
            "they",
            "them",
            "he",
            "she",
            "him",
            "her",
        }

        # Tokenize and count
        words = text.lower().split()
        words = [w.strip(".,!?;:\"'()[]{}") for w in words]
        words = [w for w in words if len(w) > 3 and w not in stopwords]

        # Get most common
        counter = Counter(words)
        return [word for word, _ in counter.most_common(top_n)]

    def group_chunks_by_source(self, chunks: list) -> Dict[str, List[Any]]:
        """Group chunks by source document.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary mapping source to chunks
        """
        grouped: Dict[str, List[Any]] = {}

        for chunk in chunks:
            metadata = self.extract_chunk_metadata(chunk)
            # Prefer document_id (original document) over source_file (may be split)
            source = (
                metadata.get("document_id")
                or metadata.get("source")
                or metadata.get("source_file")
                or "unknown"
            )

            if source not in grouped:
                grouped[source] = []

            grouped[source].append(chunk)

        return grouped

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def format_context_for_prompt(self, chunks: list, max_length: int = 3000) -> str:
        """Format chunks as context for LLM prompt.

        Args:
            chunks: List of chunks
            max_length: Maximum context length

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        context_parts = []
        current_length = 0

        for idx, chunk in enumerate(chunks, 1):
            text = self.extract_chunk_text(chunk)
            chunk_text = f"[{idx}] {text}\n"

            # Check length limit (Commandment #3: Memory management)
            if current_length + len(chunk_text) > max_length:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        return "\n".join(context_parts)

    def generate_with_llm(
        self, llm_client: Any, prompt: str, task_description: str
    ) -> str:
        """Generate content using LLM.

        Args:
            llm_client: LLM provider instance
            prompt: Prompt to send to LLM
            task_description: Description for progress display

        Returns:
            Generated text
        """
        return ProgressManager.run_with_spinner(
            lambda: self._generate_content(llm_client, prompt),
            f"Generating {task_description}...",
            f"{task_description.capitalize()} generated",
        )

    def _generate_content(self, llm_client: Any, prompt: str) -> str:
        """Generate content (internal helper).

        Args:
            llm_client: LLM provider instance
            prompt: Prompt text

        Returns:
            Generated response
        """
        # Handle different LLM provider APIs
        if hasattr(llm_client, "generate"):
            return llm_client.generate(prompt)
        elif hasattr(llm_client, "complete"):
            return llm_client.complete(prompt)
        elif callable(llm_client):
            return llm_client(prompt)
        else:
            raise TypeError(f"Unknown LLM client type: {type(llm_client)}")
