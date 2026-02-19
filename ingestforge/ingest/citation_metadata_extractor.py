#!/usr/bin/env python3
"""Backward compatibility shim for citation_metadata_extractor.

This module maintains backward compatibility by re-exporting all public APIs
from the new citation_metadata package.

DEPRECATED: Use `from ingestforge.ingest.citation_metadata import ...` instead.
This shim will be removed in a future version.
"""

# Re-export all public APIs from the new citation_metadata package
from ingestforge.ingest.citation_metadata import (
    Author,
    CitationMetadata,
    CitationMetadataExtractor,
    HTMLMetadataParser,
    PUBLISHER_DOMAIN_MAP,
    SourceType,
)

__all__ = [
    "Author",
    "CitationMetadata",
    "CitationMetadataExtractor",
    "HTMLMetadataParser",
    "PUBLISHER_DOMAIN_MAP",
    "SourceType",
]
