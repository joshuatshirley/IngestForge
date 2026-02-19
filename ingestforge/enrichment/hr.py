"""
HR and Resume enrichment.

Normalizes skills, education, and professional experience.
"""

import logging
from typing import List

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class HRMetadataRefiner:
    """
    Refines and normalizes HR-specific metadata.
    """

    # Skill normalization map (Extracted -> Normalized)
    SKILL_MAP = {
        "JS": "Javascript",
        "TS": "Typescript",
        "ML": "Machine Learning",
        "K8s": "Kubernetes",
        "AWS": "Amazon Web Services",
        "GCP": "Google Cloud Platform",
        "Azure": "Microsoft Azure",
        "React.js": "React",
        "Node": "Node.js",
    }

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich a chunk with normalized HR metadata."""
        if not chunk.metadata:
            return chunk

        raw_skills = chunk.metadata.get("resume_skills", [])
        if raw_skills:
            normalized = self._normalize_skills(raw_skills)
            chunk.metadata["resume_skills_normalized"] = normalized

            # Also add to concepts for better retrieval
            if chunk.concepts is None:
                chunk.concepts = []
            chunk.concepts = list(set(chunk.concepts + normalized))

        return chunk

    def _normalize_skills(self, skills: List[str]) -> List[str]:
        normalized = set()
        for skill in skills:
            norm_skill = self.SKILL_MAP.get(skill, skill)
            normalized.add(norm_skill)
        return sorted(list(normalized))
