#!/usr/bin/env python3
"""Citation metadata extraction for SplitAnalyze.

Extracts bibliographic metadata (author, date, publisher, DOI, ISBN, etc.)
from various document types for citation generation.

Supports:
- HTML pages (meta tags, Open Graph, Schema.org, Dublin Core)
- PDF documents (document info, XMP metadata)
- Academic identifiers (DOI, arXiv ID, PMID, ISBN)
- URL-based inference

This module has been split into focused submodules for maintainability
while maintaining 100% backward compatibility via re-exports.
"""

# Models
from ingestforge.ingest.citation_metadata.models import (
    Author,
    CitationMetadata,
    LegalMetadata,
    SourceType,
)

# Constants
from ingestforge.ingest.citation_metadata.constants import PUBLISHER_DOMAIN_MAP

# HTML Parser
from ingestforge.ingest.citation_metadata.html_parser import HTMLMetadataParser

# Extractor
from ingestforge.ingest.citation_metadata.extractors import CitationMetadataExtractor
from ingestforge.ingest.citation_metadata.legal_extractor import LegalMetadataExtractor

__all__ = [
    # Enum
    "SourceType",
    # Models
    "Author",
    "CitationMetadata",
    "LegalMetadata",
    # Constants
    "PUBLISHER_DOMAIN_MAP",
    # Classes
    "HTMLMetadataParser",
    "CitationMetadataExtractor",
    "LegalMetadataExtractor",
]
