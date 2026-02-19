#!/usr/bin/env python3
"""Citation metadata extractor for various document types.

Extracts bibliographic metadata from HTML, text, URLs, and PDF documents
for citation generation.
"""

import re
from typing import Union, Any
from urllib.parse import urlparse

from ingestforge.core.logging import get_logger
from ingestforge.ingest.citation_metadata.constants import PUBLISHER_DOMAIN_MAP
from ingestforge.ingest.citation_metadata.html_parser import HTMLMetadataParser
from ingestforge.ingest.citation_metadata.models import (
    Author,
    CitationMetadata,
    SourceType,
)

logger = get_logger(__name__)


class CitationMetadataExtractor:
    """Extract citation metadata from various sources."""

    # DOI regex pattern
    DOI_PATTERN = re.compile(r"10\.\d{4,}/[^\s]+")

    # ISBN patterns (10 or 13 digits)
    ISBN_PATTERN = re.compile(
        r"(?:ISBN[:\s-]*)?(97[89][- ]?\d{1,5}[- ]?\d+[- ]?\d+[- ]?[\dX]|\d[- ]?\d[- ]?\d[- ]?\d[- ]?\d[- ]?\d[- ]?\d[- ]?\d[- ]?\d[- ]?[\dX])",
        re.IGNORECASE,
    )

    # arXiv pattern
    ARXIV_PATTERN = re.compile(
        r"(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+/\d{7})", re.IGNORECASE
    )

    # PMID pattern
    PMID_PATTERN = re.compile(r"(?:PMID[:\s]*)(\d{7,8})", re.IGNORECASE)

    # Year pattern
    YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

    def __init__(self) -> None:
        self.metadata_cache = {}

    def _extract_basic_metadata(
        self, raw: dict[str, Any], meta: CitationMetadata
    ) -> None:
        """Extract title, authors, and date.

        Rule #4: No large functions - Extracted from extract_from_html
        """
        # Title
        meta.title = (
            raw.get("title", "") or raw.get("og_title", "") or raw.get("html_title", "")
        )

        # Authors
        authors_raw = raw.get("authors", [])
        meta.authors = [Author.from_string(a).to_dict() for a in authors_raw if a]

        # Date
        date_str = raw.get("date", "")
        meta.date_published = date_str
        if date_str:
            year_match = self.YEAR_PATTERN.search(date_str)
            if year_match:
                meta.year = int(year_match.group())

    def _extract_publication_info(
        self, raw: dict[str, Any], meta: CitationMetadata
    ) -> None:
        """Extract journal, volume, pages, publisher.

        Rule #4: No large functions - Extracted from extract_from_html
        """
        meta.journal = raw.get("journal", "")
        meta.volume = raw.get("volume", "")
        meta.issue = raw.get("issue", "")

        first_page = raw.get("first_page", "")
        last_page = raw.get("last_page", "")
        if first_page and last_page:
            meta.pages = f"{first_page}-{last_page}"
        elif first_page:
            meta.pages = first_page

        meta.publisher = raw.get("publisher", "") or raw.get("site_name", "")

    def _extract_identifiers(
        self, raw: dict[str, Any], meta: CitationMetadata, url: str
    ) -> None:
        """Extract DOI, ISBN, ISSN, etc.

        Rule #4: No large functions - Extracted from extract_from_html
        """
        meta.doi = raw.get("doi", "")
        meta.isbn = raw.get("isbn", "")
        meta.issn = raw.get("issn", "")
        meta.arxiv_id = raw.get("arxiv_id", "")
        meta.pmid = raw.get("pmid", "")
        meta.url = url or raw.get("url", "") or raw.get("canonical_url", "")

    def _extract_additional_info(
        self, raw: dict[str, Any], meta: CitationMetadata
    ) -> None:
        """Extract abstract, keywords, language.

        Rule #4: No large functions - Extracted from extract_from_html
        """
        meta.abstract = (
            raw.get("abstract", "")
            or raw.get("description", "")
            or raw.get("og_description", "")
        )
        meta.keywords = raw.get("keywords", [])
        meta.language = raw.get("language", "")

    def extract_from_html(self, html: str, url: str = "") -> CitationMetadata:
        """
        Extract metadata from HTML content.

        Rule #4: No large functions - Refactored to <60 lines

        Args:
            html: HTML content
            url: Source URL for additional inference

        Returns:
            CitationMetadata object
        """
        parser = HTMLMetadataParser()
        try:
            parser.feed(html)
        except Exception as e:
            logger.debug(f"Failed to parse HTML metadata: {e}")

        raw = parser.metadata
        meta = CitationMetadata(raw_metadata=raw, extraction_source="html")

        # Extract metadata using helper methods
        self._extract_basic_metadata(raw, meta)
        self._extract_publication_info(raw, meta)
        self._extract_identifiers(raw, meta, url)
        self._extract_additional_info(raw, meta)

        # Process JSON-LD if present
        if "json_ld" in raw:
            self._process_json_ld(raw["json_ld"], meta)

        # Infer source type and calculate confidence
        meta.source_type = self._infer_source_type(meta, url)
        meta.confidence = self._calculate_confidence(meta)

        return meta

    def _process_json_ld(
        self, ld_data: Union[dict, list], meta: CitationMetadata
    ) -> None:
        """
        Process JSON-LD structured data.

        Rule #1: Reduced nesting with helper methods
        """
        if isinstance(ld_data, list):
            for item in ld_data:
                self._process_json_ld(item, meta)
            return
        if not isinstance(ld_data, dict):
            return

        schema_type = ld_data.get("@type", "")
        if schema_type in ["Article", "NewsArticle", "BlogPosting", "ScholarlyArticle"]:
            self._process_article_schema(ld_data, meta)
        elif schema_type == "Book":
            self._process_book_schema(ld_data, meta)

    def _process_article_schema(
        self, ld_data: dict[str, Any], meta: CitationMetadata
    ) -> None:
        """
        Process Article-type JSON-LD schema.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        if not meta.title and "headline" in ld_data:
            meta.title = ld_data["headline"]

        self._extract_article_date(ld_data, meta)
        self._extract_article_authors(ld_data, meta)

        if not meta.publisher:
            publisher = ld_data.get("publisher", {})
            if isinstance(publisher, dict):
                meta.publisher = publisher.get("name", "")

        if not meta.abstract:
            meta.abstract = ld_data.get("description", "")

    def _extract_article_date(
        self, ld_data: dict[str, Any], meta: CitationMetadata
    ) -> None:
        """
        Extract date from article JSON-LD.

        Rule #1: Reduced nesting with guard clauses
        Rule #4: Function <60 lines
        """
        if meta.date_published:
            return

        date = ld_data.get("datePublished", "")
        meta.date_published = date
        if not date:
            return

        year_match = self.YEAR_PATTERN.search(date)
        if year_match:
            meta.year = int(year_match.group())

    def _extract_article_authors(
        self, ld_data: dict[str, Any], meta: CitationMetadata
    ) -> None:
        """
        Extract authors from article JSON-LD.

        Rule #1: Reduced nesting with guard clauses
        Rule #4: Function <60 lines
        """
        if meta.authors:
            return

        author = ld_data.get("author", {})

        if isinstance(author, dict):
            name = author.get("name", "")
            if name:
                meta.authors = [Author.from_string(name).to_dict()]
        elif isinstance(author, list):
            meta.authors = [
                Author.from_string(a.get("name", "")).to_dict()
                for a in author
                if a.get("name")
            ]

    def _process_book_schema(
        self, ld_data: dict[str, Any], meta: CitationMetadata
    ) -> None:
        """
        Process Book-type JSON-LD schema.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        if not meta.title and "name" in ld_data:
            meta.title = ld_data["name"]
        if not meta.isbn:
            meta.isbn = ld_data.get("isbn", "")

    def extract_from_text(self, text: str) -> CitationMetadata:
        """
        Extract identifiers and metadata from plain text.

        Args:
            text: Text content

        Returns:
            CitationMetadata with any found identifiers
        """
        meta = CitationMetadata(extraction_source="text")

        # Find DOI
        doi_match = self.DOI_PATTERN.search(text)
        if doi_match:
            meta.doi = doi_match.group()

        # Find ISBN
        isbn_match = self.ISBN_PATTERN.search(text)
        if isbn_match:
            meta.isbn = re.sub(r"[- ]", "", isbn_match.group(1))

        # Find arXiv ID
        arxiv_match = self.ARXIV_PATTERN.search(text)
        if arxiv_match:
            meta.arxiv_id = arxiv_match.group(1)

        # Find PMID
        pmid_match = self.PMID_PATTERN.search(text)
        if pmid_match:
            meta.pmid = pmid_match.group(1)

        return meta

    def extract_from_url(self, url: str) -> CitationMetadata:
        """
        Extract metadata from URL.

        Rule #4: Reduced from 61 → 40 lines (extracted publisher map)
        """
        meta = CitationMetadata(url=url, extraction_source="url")

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        for domain_key, publisher in PUBLISHER_DOMAIN_MAP.items():
            if domain_key in domain:
                meta.publisher = publisher
                break

        # Extract DOI from URL
        if "doi.org" in domain or "/doi/" in path:
            doi_match = self.DOI_PATTERN.search(url)
            if doi_match:
                meta.doi = doi_match.group()

        # Extract arXiv ID from URL
        if "arxiv.org" in domain:
            arxiv_match = self.ARXIV_PATTERN.search(path)
            if arxiv_match:
                meta.arxiv_id = arxiv_match.group(1)
            meta.source_type = SourceType.PREPRINT

        # Source type inference
        if not meta.source_type or meta.source_type == SourceType.UNKNOWN:
            meta.source_type = self._infer_source_type_from_url(domain, path)

        return meta

    def extract_from_pdf_metadata(self, pdf_info: dict[str, Any]) -> CitationMetadata:
        """
        Extract metadata from PDF document info dictionary.

        Args:
            pdf_info: PDF metadata dictionary (from PyPDF2 or similar)

        Returns:
            CitationMetadata
        """
        meta = CitationMetadata(raw_metadata=pdf_info, extraction_source="pdf")

        # Standard PDF metadata fields
        meta.title = pdf_info.get("Title", "") or pdf_info.get("/Title", "")
        author = pdf_info.get("Author", "") or pdf_info.get("/Author", "")
        if author:
            # PDF author field may contain multiple authors separated by ; or ,
            authors = re.split(r"[;,]", author)
            meta.authors = [
                Author.from_string(a.strip()).to_dict() for a in authors if a.strip()
            ]

        meta.publisher = pdf_info.get("Creator", "") or pdf_info.get("/Creator", "")

        # Date from PDF
        creation_date = pdf_info.get("CreationDate", "") or pdf_info.get(
            "/CreationDate", ""
        )
        if creation_date:
            # PDF date format: D:YYYYMMDDHHmmSS
            year_match = re.search(r"D:(\d{4})", str(creation_date))
            if year_match:
                meta.year = int(year_match.group(1))

        # Look for DOI in PDF metadata
        subject = pdf_info.get("Subject", "") or pdf_info.get("/Subject", "")
        doi_match = self.DOI_PATTERN.search(str(subject))
        if doi_match:
            meta.doi = doi_match.group()

        return meta

    def _infer_source_type(self, meta: CitationMetadata, url: str = "") -> SourceType:
        """Infer source type from metadata."""
        # If we have DOI with journal, it's likely a journal article
        if meta.doi and meta.journal:
            return SourceType.JOURNAL_ARTICLE

        # ISBN suggests a book
        if meta.isbn:
            return SourceType.BOOK

        # arXiv is preprint
        if meta.arxiv_id:
            return SourceType.PREPRINT

        # PMID suggests journal article
        if meta.pmid:
            return SourceType.JOURNAL_ARTICLE

        # URL-based inference
        if url:
            return self._infer_source_type_from_url(
                urlparse(url).netloc, urlparse(url).path
            )

        return SourceType.UNKNOWN

    def _infer_source_type_from_url(self, domain: str, path: str) -> SourceType:
        """Infer source type from URL components."""
        domain = domain.lower()
        path = path.lower()

        # Academic/preprint
        if any(
            d in domain for d in ["arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com"]
        ):
            return SourceType.PREPRINT

        # Journal/academic publisher
        if any(
            d in domain
            for d in [
                "nature.com",
                "sciencedirect.com",
                "springer.com",
                "wiley.com",
                "ieee.org",
                "acm.org",
                "plos.org",
                "pubmed.gov",
                "ncbi.nlm.nih.gov",
                "jstor.org",
            ]
        ):
            return SourceType.JOURNAL_ARTICLE

        # News
        if any(
            d in domain
            for d in [
                "nytimes.com",
                "theguardian.com",
                "bbc.com",
                "washingtonpost.com",
                "reuters.com",
                "cnn.com",
            ]
        ):
            return SourceType.NEWS_ARTICLE

        # Blog
        if (
            any(d in domain for d in ["medium.com", "wordpress.com", "blogspot.com"])
            or "/blog" in path
        ):
            return SourceType.BLOG_POST

        # GitHub/software
        if "github.com" in domain or "gitlab.com" in domain:
            return SourceType.SOFTWARE

        # Video
        if any(d in domain for d in ["youtube.com", "vimeo.com"]):
            return SourceType.VIDEO

        return SourceType.WEBPAGE

    def _calculate_confidence(self, meta: CitationMetadata) -> float:
        """Calculate confidence score based on available metadata."""
        score = 0.0
        max_score = 0.0

        # Title is essential (high weight)
        max_score += 3.0
        if meta.title:
            score += 3.0

        # Authors important
        max_score += 2.0
        if meta.authors:
            score += 2.0

        # Date/year
        max_score += 1.5
        if meta.year or meta.date_published:
            score += 1.5

        # Identifiers (DOI is most reliable)
        max_score += 2.0
        if meta.doi:
            score += 2.0
        elif meta.isbn or meta.pmid or meta.arxiv_id:
            score += 1.5

        # Publisher
        max_score += 1.0
        if meta.publisher:
            score += 1.0

        # URL
        max_score += 0.5
        if meta.url:
            score += 0.5

        return round(score / max_score, 2) if max_score > 0 else 0.0
