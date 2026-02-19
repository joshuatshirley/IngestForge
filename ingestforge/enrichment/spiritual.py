"""
Scripture and Theological enrichment.

Extracts book names, chapter/verse citations, and religious themes.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class SpiritualMetadataRefiner:
    """
    Enriches chunks with spiritual-specific metadata.
    """

    # Spiritual specific patterns (handles 1 John 1:1, Surah 2:255, etc.)
    CITATION_PATTERN = re.compile(
        r"\b((?:\d\s+)?(?:[A-Z][a-z]+))\s+(\d+[:\.]\d+(?:\-\d+)?)\b"
    )
    THEME_KEYWORDS = {
        "Grace": ["grace", "mercy", "favor"],
        "Judgment": ["judge", "judgment", "accountability"],
        "Compassion": ["compassion", "kindness", "love"],
        "Wisdom": ["wisdom", "knowledge", "understanding"],
        "Covenant": ["covenant", "promise", "oath"],
    }

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with spiritual metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Citation
        cite_match = self.CITATION_PATTERN.search(content)
        if cite_match:
            metadata["spiritual_book"] = cite_match.group(1).strip()
            metadata[
                "spiritual_citation"
            ] = f"{cite_match.group(1)} {cite_match.group(2)}"

        # Extract Themes
        found_themes = []
        for theme, keywords in self.THEME_KEYWORDS.items():
            for kw in keywords:
                if re.search(rf"\b{kw}\b", content, re.IGNORECASE):
                    found_themes.append(theme)
                    break
        if found_themes:
            metadata["spiritual_themes"] = list(set(found_themes))

        chunk.metadata = metadata
        return chunk
