"""Discovery module - Source discovery APIs for finding related content.

This module provides clients for academic paper discovery:
- ArxivSearcher: Search and download arXiv papers (raw HTTP)
- ArxivDiscovery: Search arXiv using the arxiv library (RES-002)
- SemanticScholarClient: Search with citation data
- CrossRefClient: DOI lookup and publication search
- NVDDiscovery: Search CVE vulnerabilities via NIST NVD API (CYBER-002)
- CourtListenerDiscovery: Search court opinions via CourtListener API (LEGAL-001)

Example:
    from ingestforge.discovery import ArxivSearcher, SemanticScholarClient

    arxiv = ArxivSearcher()
    papers = arxiv.search("transformer attention", limit=5)

    # Alternative using arxiv library (requires: pip install arxiv)
    from ingestforge.discovery import ArxivDiscovery
    discovery = ArxivDiscovery()
    papers = discovery.search("quantum computing", max_results=5)

    scholar = SemanticScholarClient()
    papers = scholar.search("neural networks", limit=10)

    # CVE vulnerability search (requires: pip install requests)
    from ingestforge.discovery import NVDDiscovery
    nvd = NVDDiscovery()
    cves = nvd.search("apache tomcat", max_results=10)

    # Court opinion search (LEGAL-001)
    from ingestforge.discovery import CourtListenerDiscovery
    court = CourtListenerDiscovery()
    cases = court.search("qualified immunity", jurisdiction="ca9", max_results=5)
"""

from ingestforge.discovery.arxiv_client import (
    ArxivSearcher,
    Paper,
    ArxivDownloadResult,
    SortOrder,
    SortDirection,
    export_bibtex as export_arxiv_bibtex,
)

# ArxivDiscovery uses the arxiv library (optional dependency)
# Import conditionally to avoid ImportError when arxiv is not installed
try:
    from ingestforge.discovery.arxiv_wrapper import (
        ArxivDiscovery,
        ArxivPaper,
        create_arxiv_discovery,
    )

    _ARXIV_WRAPPER_AVAILABLE = True
except ImportError:
    _ARXIV_WRAPPER_AVAILABLE = False
from ingestforge.discovery.semantic_scholar import (
    SemanticScholarClient,
    ScholarPaper,
    Author as ScholarAuthor,
    CitationResult,
    export_bibtex as export_scholar_bibtex,
)
from ingestforge.discovery.crossref import (
    CrossRefClient,
    Publication,
    Author as CrossRefAuthor,
    PublicationType,
    export_bibtex as export_crossref_bibtex,
)

# NVDDiscovery uses requests library (optional dependency)
# Import conditionally to avoid ImportError when requests is not installed
try:
    from ingestforge.discovery.nvd_wrapper import (
        NVDDiscovery,
        CVEEntry,
        Severity,
        create_nvd_discovery,
    )

    _NVD_WRAPPER_AVAILABLE = True
except ImportError:
    _NVD_WRAPPER_AVAILABLE = False

# CourtListenerDiscovery for legal case law search (LEGAL-001)
from ingestforge.discovery.courtlistener_wrapper import (
    CourtListenerDiscovery,
    CourtCase,
    PrecedentialStatus,
    FEDERAL_COURTS,
)

__all__ = [
    # arXiv (raw HTTP client)
    "ArxivSearcher",
    "Paper",
    "ArxivDownloadResult",
    "SortOrder",
    "SortDirection",
    "export_arxiv_bibtex",
    # arXiv wrapper using arxiv library (RES-002)
    "ArxivDiscovery",
    "ArxivPaper",
    "create_arxiv_discovery",
    # Semantic Scholar
    "SemanticScholarClient",
    "ScholarPaper",
    "ScholarAuthor",
    "CitationResult",
    "export_scholar_bibtex",
    # CrossRef
    "CrossRefClient",
    "Publication",
    "CrossRefAuthor",
    "PublicationType",
    "export_crossref_bibtex",
    # NVD/CVE (CYBER-002)
    "NVDDiscovery",
    "CVEEntry",
    "Severity",
    "create_nvd_discovery",
    # CourtListener (LEGAL-001)
    "CourtListenerDiscovery",
    "CourtCase",
    "PrecedentialStatus",
    "FEDERAL_COURTS",
]
