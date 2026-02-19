"""Literary metadata tagger for enriching chunks with literary analysis.

Tags chunks with literary form, verse/dramatic indicators, speaker
attribution, and section hierarchy based on structural analysis.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional

from ingestforge.ingest.literary_structure import (
    LiteraryStructureDetector,
    LiteraryStructure,
)

# Chunk type constants
CHUNK_TYPE_PRIMARY = "primary_text"
CHUNK_TYPE_CRITICISM = "criticism"
CHUNK_TYPE_WIKI_REF = "wiki_reference"
CHUNK_TYPE_REFERENCE = "reference"


class LiteraryTagger:
    """Tags chunks with literary metadata from structural analysis.

    Uses LiteraryStructureDetector to analyze full text, then applies
    literary tags (form, verse, dramatic, speakers, section hierarchy)
    to individual chunks.

    Results are cached per document to avoid redundant analysis.
    """

    def __init__(self) -> None:
        self._detector = LiteraryStructureDetector()
        self._structure_cache: Dict[str, LiteraryStructure] = {}

    def tag_chunks(
        self,
        chunks: list[Any],
        full_text: str,
        chunk_type: str = CHUNK_TYPE_PRIMARY,
        document_id: Optional[str] = None,
    ) -> list[Any]:
        """
        Tag chunks with literary metadata from structural analysis.

        Rule #4: Reduced from 75 lines to <60 lines via helper extraction

        Modifies chunks in place and returns the same list.

        Args:
            chunks: List of ChunkRecord objects to tag.
            full_text: The full source text (used for structure analysis).
            chunk_type: The chunk_type to set on all chunks.
            document_id: Optional document ID for caching.

        Returns:
            The same list of chunks, now enriched with literary metadata.
        """
        if not chunks:
            return chunks
        structure = self._compute_or_get_structure(full_text, document_id)
        speakers = self._extract_speakers(structure)
        for chunk in chunks:
            self._tag_single_chunk(chunk, structure, speakers, chunk_type, full_text)

        return chunks

    def _compute_or_get_structure(
        self, full_text: str, document_id: Optional[str]
    ) -> Any:
        """
        Compute or retrieve cached structure analysis.

        Rule #4: Extracted to reduce function size
        """
        cache_key = (
            document_id
            or hashlib.md5(full_text.encode("utf-8", errors="replace")).hexdigest()
        )

        if cache_key not in self._structure_cache:
            self._structure_cache[cache_key] = self._detector.analyze(full_text)

        return self._structure_cache[cache_key]

    def _extract_speakers(self, structure: Any) -> list[Any]:
        """
        Extract unique speakers from structure dialogue.

        Rule #4: Extracted to reduce function size
        """
        return sorted(set(d.speaker for d in structure.dialogue))

    def _tag_single_chunk(
        self,
        chunk: Any,
        structure: Any,
        speakers: list[Any],
        chunk_type: str,
        full_text: str,
    ) -> None:
        """
        Tag a single chunk with literary metadata.

        Rule #4: Extracted to reduce function size
        """
        # Set chunk type
        chunk.chunk_type = chunk_type

        # Initialize concepts if None
        if chunk.concepts is None:
            chunk.concepts = []
        self._add_structural_concepts(chunk, structure)
        self._tag_speakers(chunk, speakers)
        self._set_section_info(chunk, full_text)

    def _add_structural_concepts(self, chunk: Any, structure: Any) -> None:
        """
        Add structural literary concepts to chunk.

        Rule #4: Extracted to reduce function size
        """
        # Add form concept
        form_tag = f"form:{structure.estimated_form}"
        if form_tag not in chunk.concepts:
            chunk.concepts.append(form_tag)

        # Add verse concept
        if structure.is_verse and "verse" not in chunk.concepts:
            chunk.concepts.append("verse")

        # Add dramatic concept
        if structure.is_dramatic and "dramatic" not in chunk.concepts:
            chunk.concepts.append("dramatic")

    def _set_section_info(self, chunk: Any, full_text: str) -> None:
        """
        Set section hierarchy information on chunk.

        Rule #4: Extracted to reduce function size
        """
        content = getattr(chunk, "content", "")
        if not content or not full_text:
            return

        # Find chunk position in full text
        prefix = content[:100]
        pos = full_text.find(prefix)
        if pos < 0:
            return

        hierarchy = self._detector.get_section_hierarchy(full_text, pos)
        chunk.section_hierarchy = hierarchy
        if hierarchy:
            chunk.section_title = hierarchy[-1]

    def tag_reference_chunks(
        self,
        chunks: list[Any],
        chunk_type: str = CHUNK_TYPE_WIKI_REF,
        source_tag: Optional[str] = None,
    ) -> list[Any]:
        """Tag chunks as reference material.

        Sets chunk_type and optionally adds a source tag (e.g., "source:wikipedia").

        Args:
            chunks: List of ChunkRecord objects.
            chunk_type: The reference type to set.
            source_tag: Optional source identifier to add as concept.

        Returns:
            The same list of chunks, now tagged.
        """
        for chunk in chunks:
            chunk.chunk_type = chunk_type

            if source_tag:
                if chunk.concepts is None:
                    chunk.concepts = []
                tag = f"source:{source_tag}"
                if tag not in chunk.concepts:
                    chunk.concepts.append(tag)

        return chunks

    def _tag_speakers(self, chunk: Any, speakers: List[str]) -> None:
        """Add speaker tags for speakers mentioned in this chunk's content."""
        content = getattr(chunk, "content", "")
        if not content or not speakers:
            return

        content_lower = content.lower()
        tagged = []

        for speaker in speakers:
            # Use word boundary matching to avoid false positives
            # e.g., "Alice" should not match "palace"
            pattern = re.compile(
                r"\b" + re.escape(speaker) + r"\b",
                re.IGNORECASE,
            )
            if pattern.search(content):
                tag = f"speaker:{speaker}"
                if tag not in chunk.concepts:
                    tagged.append(tag)

        # Add sorted speaker tags
        for tag in sorted(tagged):
            chunk.concepts.append(tag)
