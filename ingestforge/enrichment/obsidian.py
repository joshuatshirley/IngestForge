"""
Obsidian-specific metadata extraction and enrichment.

Maps Obsidian vault metadata (tags, aliases, properties) to IngestForge
chunk metadata and concepts.
"""

import logging
from typing import List, Dict, Any

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class ObsidianMetadataRefiner:
    """
    Enriches chunks with Obsidian-specific metadata.
    """

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich a chunk with Obsidian metadata from frontmatter."""
        # Check if we have frontmatter in metadata
        if not chunk.metadata:
            return chunk

        # Get frontmatter if nested, otherwise use the whole metadata dict
        frontmatter = chunk.metadata.get("frontmatter")
        if not isinstance(frontmatter, dict):
            frontmatter = chunk.metadata

        # Extract tags
        tags = self._extract_tags(frontmatter)
        if tags:
            if chunk.concepts is None:
                chunk.concepts = []
            # Merge unique tags into concepts
            chunk.concepts = list(set(chunk.concepts + tags))
            chunk.metadata["obsidian_tags"] = tags

        # Extract aliases
        aliases = self._extract_aliases(frontmatter)
        if aliases:
            chunk.metadata["aliases"] = aliases

        return chunk

    def _extract_tags(self, frontmatter: Dict[str, Any]) -> List[str]:
        """Extract tags from frontmatter (string or list)."""
        tags_raw = frontmatter.get("tags") or frontmatter.get("tag", [])
        if isinstance(tags_raw, str):
            # Split by comma or space if it's a string
            return [
                t.strip().lstrip("#")
                for t in tags_raw.replace(",", " ").split()
                if t.strip()
            ]
        if isinstance(tags_raw, list):
            return [str(t).lstrip("#") for t in tags_raw]
        return []

    def _extract_aliases(self, frontmatter: Dict[str, Any]) -> List[str]:
        """Extract aliases from frontmatter."""
        aliases = frontmatter.get("aliases") or frontmatter.get("alias", [])
        if isinstance(aliases, str):
            return [a.strip() for a in aliases.split(",") if a.strip()]
        if isinstance(aliases, list):
            return [str(a) for a in aliases]
        return []
