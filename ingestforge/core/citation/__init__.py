"""Citation utilities package.

Provides citation export and formatting functionality:
- export_formats: BibTeX and RIS format exporters (CITE-002.1)
- link_extractor: Internal reference extractor (CITE-001.1)
- analytics: Graph analytics engine (CITE-001.2)
"""

from __future__ import annotations

from ingestforge.core.citation.export_formats import (
    Citation,
    BibTeXFormatter,
    RISFormatter,
    format_bibtex,
    format_ris,
    export_bibliography,
)

from ingestforge.core.citation.link_extractor import (
    ReferenceType,
    InternalReference,
    LinkMap,
    LinkExtractor,
    extract_internal_references,
)

from ingestforge.core.citation.analytics import (
    NodeRole,
    NodeMetrics,
    GraphStats,
    GraphAnalytics,
)

__all__ = [
    # Export formats (CITE-002.1)
    "Citation",
    "BibTeXFormatter",
    "RISFormatter",
    "format_bibtex",
    "format_ris",
    "export_bibliography",
    # Link extractor (CITE-001.1)
    "ReferenceType",
    "InternalReference",
    "LinkMap",
    "LinkExtractor",
    "extract_internal_references",
    # Analytics (CITE-001.2)
    "NodeRole",
    "NodeMetrics",
    "GraphStats",
    "GraphAnalytics",
]
