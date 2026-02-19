#!/usr/bin/env python3
"""Legal metadata extractor for case law and court documents.

Extracts legal-specific metadata like docket numbers, jurisdictions, judges,
and court reporters from documents.
"""

import re
from typing import Any, Dict

from ingestforge.core.logging import get_logger
from ingestforge.ingest.citation_metadata.extractors import CitationMetadataExtractor
from ingestforge.ingest.citation_metadata.models import LegalMetadata, SourceType

logger = get_logger(__name__)


class LegalMetadataExtractor(CitationMetadataExtractor):
    """Specialized extractor for legal documents."""

    # Docket number patterns (e.g., 20-1234, No. 12-345, 1:22-cv-00001)
    DOCKET_PATTERNS = [
        re.compile(r"\b(?:Case|Matter)\s*No\.\s*([\w\d\-\:]+)\b", re.IGNORECASE),
        re.compile(r"\bDocket\s*No\.\s*([\d\-]+)\b", re.IGNORECASE),
        re.compile(r"\bNo\.\s*([\d\-]+)\b", re.IGNORECASE),
        re.compile(
            r"\b(\d{1,2}-[a-zA-Z]{1,4}-\d{3,7}-[a-zA-Z]{1,4})\b"
        ),  # e.g., 11-cv-01846-LHK
        re.compile(r"\b(\d{1,2}-\d{3,7})\b"),  # Simple 20-1234 format
    ]

    # Jurisdiction patterns
    JURISDICTION_PATTERNS = [
        re.compile(r"\b(?:United\s+States|U\.S\.)\s+Supreme\s+Court\b", re.IGNORECASE),
        re.compile(
            r"\b(?:United\s+States|U\.S\.)\s+Court\s+of\s+Appeals\s+for\s+the\s+(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|District\s+of\s+Columbia|Federal)\s+Circuit\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:United\s+States|U\.S\.)\s+District\s+Court\b", re.IGNORECASE),
        re.compile(r"\b(?:United\s+States|U\.S\.)\s+Tax\s+Court\b", re.IGNORECASE),
    ]

    # Judge patterns (e.g., Judge Smith, Justice Scalia, Judge: Lucy H. Koh)
    JUDGE_PATTERNS = [
        re.compile(r"\b(?:Chief\s+)?Judge[:\s]+([A-Z][a-z\.]+(?:\s+[A-Z][a-z\.]+)*)\b"),
        re.compile(r"\bJustice[:\s]+([A-Z][a-z\.]+(?:\s+[A-Z][a-z\.]+)*)\b"),
    ]

    # Reporter patterns (Bluebook style)
    REPORTER_PATTERNS = [
        re.compile(
            r"\b(\d+)\s+(U\.S\.|S\.\s*Ct\.|L\.\s*Ed\.\s*2d|F\.\s*(?:2d|3d)?|F\.\s*Supp\.\s*(?:2d|3d)?)\s+(\d+)\b"
        ),
    ]

    def extract_from_json(self, data: Dict[str, Any]) -> LegalMetadata:
        """Extract legal metadata from a JSON payload (e.g., CourtListener)."""
        raw_metadata = data.copy()
        meta = LegalMetadata(
            raw_metadata=raw_metadata,
            extraction_source="json_legal",
            source_type=SourceType.COURT_OPINION,
        )

        # Basic metadata from CourtListener format
        meta.title = data.get("case_name", "") or data.get("title", "")
        meta.docket_number = data.get("docket_number", "")
        meta.court = data.get("court", "")
        meta.jurisdiction = data.get("jurisdiction", "")
        meta.judge = data.get("judge", "") or data.get("author_name", "")
        meta.date_published = data.get("date_filed", "") or data.get(
            "date_published", ""
        )

        if meta.date_published:
            year_match = self.YEAR_PATTERN.search(meta.date_published)
            if year_match:
                meta.year = int(year_match.group())

        # Citation handling
        meta.citation = data.get("citation", "") or data.get("neutral_citation", "")

        meta.confidence = self._calculate_legal_confidence(meta)
        return meta

    def extract_from_text(self, text: str) -> LegalMetadata:
        """Extract legal identifiers from plain text."""
        # Start with basic extraction
        base_meta = super().extract_from_text(text)

        # Merge basic metadata into LegalMetadata
        meta_dict = base_meta.to_dict()
        meta_dict.pop("extraction_source", None)
        meta_dict.pop("source_type", None)

        meta = LegalMetadata(
            **meta_dict,
            extraction_source="text_legal",
            source_type=SourceType.COURT_OPINION,
        )

        # Find Year
        year_match = self.YEAR_PATTERN.search(text)
        if year_match:
            meta.year = int(year_match.group())

        # Find Docket Number
        for pattern in self.DOCKET_PATTERNS:
            match = pattern.search(text)
            if match:
                meta.docket_number = match.group(1)
                break

        # Find Jurisdiction
        for pattern in self.JURISDICTION_PATTERNS:
            match = pattern.search(text)
            if match:
                meta.jurisdiction = match.group(0)
                break

        # Find Judge
        for pattern in self.JUDGE_PATTERNS:
            match = pattern.search(text)
            if match:
                meta.judge = match.group(1)
                break

        # Find Reporter/Citation
        for pattern in self.REPORTER_PATTERNS:
            match = pattern.search(text)
            if match:
                meta.reporter = match.group(2)
                meta.citation = match.group(0)
                break

        meta.confidence = self._calculate_legal_confidence(meta)
        return meta

    def _calculate_legal_confidence(self, meta: LegalMetadata) -> float:
        """Calculate confidence score specialized for legal documents."""
        score = 0.0
        max_score = 0.0

        # Base confidence from parent
        score += super()._calculate_confidence(meta) * 5.0
        max_score += 5.0

        # Legal specific boosts
        max_score += 2.0
        if meta.docket_number:
            score += 2.0

        max_score += 1.0
        if meta.jurisdiction or meta.court:
            score += 1.0

        max_score += 1.0
        if meta.judge:
            score += 1.0

        max_score += 2.0
        if meta.citation or meta.reporter:
            score += 2.0

        return round(score / max_score, 2) if max_score > 0 else 0.0
