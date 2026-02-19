"""
Educational and Pedagogical enrichment.

Extracts grade levels, subjects, and standards from educational materials.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class EduMetadataRefiner:
    """
    Enriches chunks with educational-specific metadata.
    """

    # Edu specific patterns
    GRADE_PATTERN = re.compile(
        r"\b(?:Grade|Level)[:\s]+(\d{1,2}|[Kk]|Kindergarten|Primary|Secondary|Higher Ed)\b",
        re.IGNORECASE,
    )
    SUBJECT_PATTERN = re.compile(
        r"\b(?:Subject|Topic|Course)[:\s]+([\w\s\-]{3,40})(?=\n|\.|\Z)", re.IGNORECASE
    )
    STANDARD_PATTERN = re.compile(
        r"\b((?:CCSS|NGSS|ISTE|TEKS)[A-Z0-9\.\-]+)\b", re.IGNORECASE
    )

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with educational metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Grade Level
        grade_match = self.GRADE_PATTERN.search(content)
        if grade_match:
            metadata["edu_grade_level"] = grade_match.group(1).strip().capitalize()

        # Extract Subject
        subject_match = self.SUBJECT_PATTERN.search(content)
        if subject_match:
            metadata["edu_subject"] = subject_match.group(1).strip()

        # Extract Standards
        standards = self.STANDARD_PATTERN.findall(content)
        if standards:
            metadata["edu_standards"] = list(set([s.upper() for s in standards]))

        chunk.metadata = metadata
        return chunk
