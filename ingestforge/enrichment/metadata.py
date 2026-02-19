"""
Metadata extraction for chunks.

Extract additional metadata from chunk content.
"""

import re
from typing import Any, Dict, List

from ingestforge.chunking.semantic_chunker import ChunkRecord


class MetadataExtractor:
    """
    Extract metadata from chunk content.

    Extracts:
    - Keywords/key phrases
    - Numbers and statistics
    - Dates
    - Named references
    """

    def __init__(self) -> None:
        # Common patterns
        self.date_pattern = re.compile(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b|"
            r"\b\d{4}[/-]\d{2}[/-]\d{2}\b",
            re.IGNORECASE,
        )
        self.number_pattern = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )

    def extract(self, chunk: ChunkRecord) -> Dict[str, Any]:
        """
        Extract metadata from a chunk.

        Args:
            chunk: Chunk to analyze

        Returns:
            Dictionary of extracted metadata
        """
        content = chunk.content

        metadata = {
            "keywords": self._extract_keywords(content),
            "dates": self._extract_dates(content),
            "numbers": self._extract_numbers(content),
            "urls": self._extract_urls(content),
            "emails": self._extract_emails(content),
            "has_list": self._has_list(content),
            "has_headers": self._has_headers(content),
            "paragraph_count": content.count("\n\n") + 1,
        }

        return metadata

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """
        Add extracted metadata to chunk concepts field.

        Args:
            chunk: Chunk to enrich

        Returns:
            Chunk with concepts populated
        """
        metadata = self.extract(chunk)

        # Store keywords as concepts
        if metadata["keywords"]:
            chunk.concepts = metadata["keywords"]

        return chunk

    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key phrases using simple TF-based approach."""
        # Tokenize
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())

        # Common stop words
        stop_words = {
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
            "has",
            "have",
            "been",
            "would",
            "could",
            "should",
            "their",
            "there",
            "this",
            "that",
            "with",
            "they",
            "from",
            "will",
            "what",
            "when",
            "where",
            "which",
            "also",
            "more",
            "some",
            "than",
            "into",
            "very",
            "just",
            "only",
        }

        # Count words
        word_counts: Dict[str, int] = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Get top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        return [word for word, _ in sorted_words[:top_n]]

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        matches = self.date_pattern.findall(text)
        return list(set(matches))[:10]

    def _extract_numbers(self, text: str) -> List[str]:
        """Extract significant numbers from text."""
        matches = self.number_pattern.findall(text)
        # Filter out very short numbers (likely not significant)
        significant = [n for n in matches if len(n) >= 3 or "%" in n]
        return list(set(significant))[:20]

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        matches = self.url_pattern.findall(text)
        return list(set(matches))[:10]

    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text."""
        matches = self.email_pattern.findall(text)
        return list(set(matches))[:10]

    def _has_list(self, text: str) -> bool:
        """Check if text contains a list."""
        list_patterns = [
            r"^\s*[-*•]\s",  # Bullet list
            r"^\s*\d+\.\s",  # Numbered list
            r"^\s*[a-z]\)\s",  # Letter list
        ]
        for pattern in list_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def _has_headers(self, text: str) -> bool:
        """Check if text contains headers."""
        # Markdown headers
        if re.search(r"^#+\s", text, re.MULTILINE):
            return True
        # ALL CAPS lines (potential headers)
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if len(line) > 5 and len(line) < 100 and line.isupper():
                return True
        return False
