"""
Web search integration for research sessions.

Provides DuckDuckGo-based web search with educational domain boosting,
relevance scoring, and domain filtering. Used by the research orchestrator
to discover sources for a research topic.

Requires: pip install duckduckgo-search  (or pip install ingestforge[research])
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from ingestforge.core.config import WebSearchConfig


@dataclass
class SearchResult:
    """A single web search result."""

    url: str
    title: str
    snippet: str
    domain: str = ""
    relevance_score: float = 0.0
    is_educational: bool = False

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc.lower()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        return cls(
            **{
                k: v
                for k, v in data.items()
                if k
                in {
                    "url",
                    "title",
                    "snippet",
                    "domain",
                    "relevance_score",
                    "is_educational",
                }
            }
        )


@dataclass
class SearchSession:
    """A record of a web search performed."""

    query: str
    results: List[SearchResult] = field(default_factory=list)
    searched_at: str = ""
    total_found: int = 0
    filters_applied: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.searched_at:
            self.searched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "searched_at": self.searched_at,
            "total_found": self.total_found,
            "filters_applied": self.filters_applied,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchSession":
        results = [SearchResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            query=data["query"],
            results=results,
            searched_at=data.get("searched_at", ""),
            total_found=data.get("total_found", 0),
            filters_applied=data.get("filters_applied", {}),
        )


class WebSearcher:
    """
    Web search via DuckDuckGo with educational boosting and domain filtering.

    Uses the duckduckgo-search library (no API key needed).
    """

    # Default educational TLDs and domains for scoring
    EDUCATIONAL_TLDS = {".edu", ".gov", ".ac.uk", ".edu.au"}
    EDUCATIONAL_DOMAINS = {
        "wikipedia.org",
        "arxiv.org",
        "scholar.google.com",
        "pubmed.ncbi.nlm.nih.gov",
        "jstor.org",
        "ncbi.nlm.nih.gov",
        "nature.com",
        "sciencedirect.com",
        "springer.com",
        "wiley.com",
        "ieee.org",
        "acm.org",
    }

    def __init__(self, config: Optional[WebSearchConfig] = None) -> None:
        self.config = config or WebSearchConfig()

    def _import_ddgs(self):
        """
        Import DDGS package with fallback.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Returns:
            DDGS class

        Raises:
            ImportError: If neither package is installed
        """
        try:
            # Try new package name first (ddgs), fallback to old (duckduckgo_search)
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            return DDGS
        except ImportError:
            raise ImportError(
                "ddgs is required for web search. "
                "Install with: pip install ddgs  "
                "or: pip install ingestforge[research]"
            )

    def _execute_ddgs_search(
        self, query: str, region: str, max_results: int
    ) -> list[Any]:
        """
        Execute DDGS search and return raw results.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            query: Search query
            region: Search region
            max_results: Maximum results

        Returns:
            List of raw search result dictionaries

        Raises:
            RuntimeError: If search fails
        """
        DDGS = self._import_ddgs()
        try:
            ddgs = DDGS()
            return list(
                ddgs.text(
                    query,
                    region=region,
                    safesearch=self.config.safe_search,
                    max_results=max_results,
                )
            )
        except Exception as e:
            raise RuntimeError(f"Web search failed: {e}") from e

    def _convert_raw_result(self, raw: dict) -> SearchResult:
        """
        Convert raw search result to SearchResult.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            raw: Raw result dictionary from DDGS

        Returns:
            SearchResult object
        """
        url = raw.get("href", raw.get("url", ""))
        title = raw.get("title", "")
        snippet = raw.get("body", raw.get("snippet", ""))

        return SearchResult(
            url=url,
            title=title,
            snippet=snippet,
        )

    def _process_and_score_results(self, raw_results: list[Any]) -> list[Any]:
        """
        Process and score all search results.

        Rule #1: Early continue for filtering
        Rule #2: Fixed loop bound
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            raw_results: List of raw result dictionaries

        Returns:
            List of processed and scored SearchResult objects
        """
        results = []
        for i, raw in enumerate(raw_results):
            result = self._convert_raw_result(raw)
            if self._is_excluded(result.domain):
                continue

            # Score the result
            result.is_educational = self._is_educational(result.domain)
            result.relevance_score = self._score_result(result, position=i)

            results.append(result)

        return results

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: Optional[str] = None,
    ) -> SearchSession:
        """
        Search the web for a query.

        Rule #1: Early validation with exception
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            query: Search query string.
            max_results: Max results to return (default from config).
            region: Search region (default from config).

        Returns:
            SearchSession with scored and sorted results.

        Raises:
            ValueError: If query is empty.
            ImportError: If duckduckgo-search is not installed.
            RuntimeError: If search fails.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        max_results = max_results or self.config.max_results
        region = region or self.config.region

        # Execute search using helpers
        raw_results = self._execute_ddgs_search(query, region, max_results)
        results = self._process_and_score_results(raw_results)

        # Sort by relevance score descending
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        return SearchSession(
            query=query,
            results=results,
            total_found=len(raw_results),
            filters_applied={
                "region": region,
                "safe_search": self.config.safe_search,
                "excluded_domains": self.config.excluded_domains,
                "educational_boost": self.config.educational_boost,
            },
        )

    def search_academic(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> SearchSession:
        """
        Search with academic/educational site filters.

        Appends site: operators for educational domains.
        """
        # Build a query that targets educational sites
        site_filters = []
        for domain in self.config.educational_domains[:5]:
            # Clean domain - remove leading dots for site: operator
            clean = domain.lstrip(".")
            site_filters.append(f"site:{clean}")

        academic_query = f"{query} ({' OR '.join(site_filters)})"
        return self.search(academic_query, max_results=max_results)

    def _score_result(self, result: SearchResult, position: int) -> float:
        """
        Score a search result based on position and domain authority.

        Scoring:
        - Position decay: starts at 1.0, decays by 0.03 per position
        - Educational domain bonus: +0.15 for .edu/.gov, +0.10 for known academic sites
        """
        # Position-based score (higher rank = higher score)
        score = max(0.1, 1.0 - (position * 0.03))

        # Educational boost
        if self.config.educational_boost and result.is_educational:
            domain = result.domain.lower()
            # Strong boost for .edu and .gov
            if any(domain.endswith(tld) for tld in self.EDUCATIONAL_TLDS):
                score += 0.15
            else:
                # Moderate boost for known academic domains
                score += 0.10

        return round(min(score, 1.5), 4)

    def _is_educational(self, domain: str) -> bool:
        """Check if a domain is educational/academic."""
        domain = domain.lower()

        # Check TLDs
        for tld in self.EDUCATIONAL_TLDS:
            if domain.endswith(tld):
                return True

        # Check known educational domains
        for edu_domain in self.EDUCATIONAL_DOMAINS:
            if domain == edu_domain or domain.endswith("." + edu_domain):
                return True

        # Check config educational domains
        for edu_domain in self.config.educational_domains:
            clean = edu_domain.lstrip(".")
            if (
                domain == clean
                or domain.endswith("." + clean)
                or domain.endswith(edu_domain)
            ):
                return True

        return False

    def _is_excluded(self, domain: str) -> bool:
        """Check if a domain is in the exclusion list."""
        domain = domain.lower()
        for excluded in self.config.excluded_domains:
            excluded = excluded.lower()
            if domain == excluded or domain.endswith("." + excluded):
                return True
        return False
