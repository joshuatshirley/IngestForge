"""
HTML document processor for IngestForge.

Extracts article content, metadata, and structure from HTML files
using trafilatura for smart content extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ingestforge.ingest.extraction_rules import ExtractionRuleRegistry

from ingestforge.core.provenance import (
    SourceLocation,
    SourceType,
    Author,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HTMLSection:
    """A section within an HTML document."""

    level: int  # 1 for h1, 2 for h2, etc.
    title: str
    content: str
    subsections: List["HTMLSection"] = field(default_factory=list)


@dataclass
class ExtractedHTML:
    """Result of HTML extraction."""

    # Content
    text: str  # Clean text content
    markdown: str  # Markdown formatted content
    html_clean: str  # Cleaned HTML

    # Metadata
    title: str
    authors: List[str]
    publication_date: Optional[str]
    description: Optional[str]
    site_name: Optional[str]
    url: Optional[str]
    language: Optional[str]

    # Structure
    sections: List[HTMLSection]
    headings: List[Dict[str, Any]]  # [{"level": 2, "text": "Introduction"}, ...]

    # Provenance
    source_location: SourceLocation

    # Raw data
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


class HTMLProcessor:
    """
    Process HTML files to extract content and metadata.

    Uses trafilatura for intelligent article extraction and
    BeautifulSoup for structure parsing. Supports configurable
    extraction rules for site-specific content extraction.

    Example:
        processor = HTMLProcessor()
        result = processor.process(Path("article.html"))
        print(result.title)
        print(result.markdown)
        print(result.source_location.to_short_cite())

        # With extraction rules
        from ingestforge.ingest.extraction_rules import create_default_registry
        processor = HTMLProcessor(extraction_registry=create_default_registry())
        result = processor.process_url("https://en.wikipedia.org/wiki/Python")
    """

    def __init__(
        self,
        include_tables: bool = True,
        include_links: bool = True,
        include_images: bool = False,
        favor_precision: bool = True,
        extraction_registry: Optional["ExtractionRuleRegistry"] = None,
        rules_dir: Optional[Path] = None,
    ):
        """
        Initialize HTML processor.

        Args:
            include_tables: Extract tables from content
            include_links: Preserve links in output
            include_images: Include image references
            favor_precision: Prefer precision over recall in extraction
            extraction_registry: Optional pre-configured ExtractionRuleRegistry
            rules_dir: Optional directory containing extraction rule files
        """
        self.include_tables = include_tables
        self.include_links = include_links
        self.include_images = include_images
        self.favor_precision = favor_precision
        self._extraction_registry = extraction_registry
        self._rules_dir = rules_dir

    @property
    def extraction_registry(self) -> Optional["ExtractionRuleRegistry"]:
        """
        Lazy-load extraction rule registry.

        Returns None if no registry or rules_dir was configured.
        """
        if self._extraction_registry is None and self._rules_dir:
            self._extraction_registry = self._load_extraction_registry()
        return self._extraction_registry

    def _load_extraction_registry(self) -> Optional["ExtractionRuleRegistry"]:
        """Load extraction registry from rules directory."""
        try:
            from ingestforge.ingest.extraction_rules import ExtractionRuleRegistry

            registry = ExtractionRuleRegistry(rules_dir=self._rules_dir)
            return registry if registry.rules else None
        except Exception as e:
            logger.debug(f"Failed to load extraction registry: {e}")
            return None

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the file."""
        suffix = file_path.suffix.lower()
        return suffix in [".html", ".htm", ".mhtml", ".xhtml"]

    def process(
        self,
        file_path: Path,
        source_url: Optional[str] = None,
    ) -> ExtractedHTML:
        """
        Process an HTML file.

        Args:
            file_path: Path to HTML file
            source_url: Original URL if known

        Returns:
            ExtractedHTML with content, metadata, and provenance
        """
        # Read and extract content
        html_content = self._read_file(file_path)

        # Apply extraction rule if available for this URL
        rule_extracted = self._apply_extraction_rule(html_content, source_url)

        if rule_extracted:
            extracted, rule_metadata = rule_extracted
            metadata = self._extract_metadata(html_content, source_url)
            metadata.update(rule_metadata)
        else:
            extracted = self._extract_content(html_content, source_url)
            metadata = self._extract_metadata(html_content, source_url)

        # Enrich with citation metadata
        self._enrich_citation_metadata(html_content, source_url, metadata)

        # Extract structure
        sections, headings = self._extract_structure(html_content)

        # Add tables if enabled
        if self.include_tables:
            self._append_table_content(html_content, extracted)

        # Build source location
        source_location = self._build_source_location(
            file_path=file_path,
            url=source_url or metadata.get("url"),
            title=metadata.get("title", file_path.stem),
            authors=metadata.get("authors", []),
            publication_date=metadata.get("date"),
        )

        return self._build_extracted_html(
            extracted, metadata, sections, headings, source_location, source_url
        )

    def _apply_extraction_rule(
        self,
        html_content: str,
        url: Optional[str],
    ) -> Optional[tuple[Dict[str, str], Dict[str, Any]]]:
        """
        Apply site-specific extraction rule if available.

        Args:
            html_content: Raw HTML content.
            url: Source URL (used to match rules).

        Returns:
            Tuple of (extracted_content, metadata) if rule matched, None otherwise.
        """
        if not url or not self.extraction_registry:
            return None

        rule = self.extraction_registry.find_matching_rule(url)
        if not rule:
            return None

        try:
            return self._extract_with_rule(html_content, rule)
        except Exception as e:
            logger.debug(f"Rule-based extraction failed for {rule.name}: {e}")
            return None

    def _extract_with_rule(
        self,
        html_content: str,
        rule: Any,
    ) -> Optional[tuple[Dict[str, str], Dict[str, Any]]]:
        """
        Extract content using a specific extraction rule.

        Args:
            html_content: Raw HTML content.
            rule: ExtractionRule to apply.

        Returns:
            Tuple of (extracted_content, metadata) or None if extraction fails.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return None

        soup = BeautifulSoup(html_content, "lxml")

        # Apply content boundaries if defined
        if rule.boundaries and rule.boundaries.container:
            content_area = self._apply_content_boundaries(soup, rule.boundaries)
        else:
            content_area = soup

        if content_area is None:
            return None

        # Extract fields defined in the rule
        metadata = self._extract_rule_fields(content_area, rule)

        # Extract text content from the bounded area
        text = content_area.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return (
            {"text": text, "markdown": text, "html_clean": str(content_area)},
            metadata,
        )

    def _apply_content_boundaries(
        self,
        soup: Any,
        boundaries: Any,
    ) -> Optional[Any]:
        """
        Apply content boundary rules to extract main content area.

        Args:
            soup: BeautifulSoup object.
            boundaries: ContentBoundary with container and removal selectors.

        Returns:
            BeautifulSoup element representing the content area.
        """
        # Find container
        container = self._find_element(soup, boundaries.container)
        if container is None:
            return soup

        # Remove unwanted elements
        for remove_selector in boundaries.remove:
            for element in self._find_all_elements(container, remove_selector):
                element.decompose()

        return container

    def _find_element(self, soup: Any, selector: Any) -> Optional[Any]:
        """Find single element using selector."""
        if selector.type.value == "css":
            return soup.select_one(selector.value)
        # XPath not directly supported by BeautifulSoup
        return None

    def _find_all_elements(self, soup: Any, selector: Any) -> List[Any]:
        """Find all elements matching selector."""
        if selector.type.value == "css":
            return soup.select(selector.value)
        return []

    def _extract_rule_fields(self, content: Any, rule: Any) -> Dict[str, Any]:
        """
        Extract metadata fields defined in the rule.

        Args:
            content: BeautifulSoup element to extract from.
            rule: ExtractionRule with field definitions.

        Returns:
            Dictionary of extracted field values.
        """
        metadata: Dict[str, Any] = {}

        for field in rule.fields:
            value = self._extract_field_value(content, field)
            if value is not None:
                metadata[field.name] = value
            elif field.default is not None:
                metadata[field.name] = field.default

        return metadata

    def _extract_field_value(self, content: Any, field: Any) -> Optional[Any]:
        """
        Extract a single field value from content.

        Args:
            content: BeautifulSoup element.
            field: ExtractionField definition.

        Returns:
            Extracted value or None.
        """
        elements = self._find_all_elements(content, field.selector)
        if not elements:
            return None

        if field.selector.multiple:
            values = [self._get_element_value(el, field) for el in elements]
            return [v for v in values if v]

        return self._get_element_value(elements[0], field)

    def _get_element_value(self, element: Any, field: Any) -> Optional[str]:
        """Get value from element based on extraction strategy."""
        strategy = field.strategy.value

        if strategy == "attribute" and field.attribute:
            value = element.get(field.attribute, "")
        elif strategy == "html":
            value = str(element)
        elif strategy == "all_text":
            value = element.get_text(separator=" ", strip=True)
        else:  # text or markdown
            value = element.get_text(strip=True)

        # Apply filters
        if value and field.filters:
            value = self._apply_filters(value, field.filters)

        return value if value else None

    def _apply_filters(self, value: str, filters: List[str]) -> str:
        """Apply filter chain to extracted value."""
        if not self.extraction_registry:
            return value

        for filter_name in filters:
            if filter_name in self.extraction_registry._filters:
                value = self.extraction_registry._filters[filter_name](value)

        return value

    def _enrich_citation_metadata(
        self, html_content: str, source_url: Optional[str], metadata: dict[str, Any]
    ):
        """Enrich metadata with citation identifiers (DOI, arXiv, etc.)."""
        try:
            from ingestforge.ingest.citation_metadata_extractor import (
                CitationMetadataExtractor,
            )

            citation_extractor = CitationMetadataExtractor()
            citation_meta = citation_extractor.extract_from_html(
                html_content, url=source_url or ""
            )

            if citation_meta.doi:
                metadata["doi"] = citation_meta.doi
            if citation_meta.arxiv_id:
                metadata["arxiv_id"] = citation_meta.arxiv_id
            if citation_meta.isbn:
                metadata["isbn"] = citation_meta.isbn
            if citation_meta.pmid:
                metadata["pmid"] = citation_meta.pmid
            if citation_meta.journal:
                metadata["journal"] = citation_meta.journal
            if citation_meta.abstract:
                metadata["abstract"] = citation_meta.abstract
        except Exception as e:
            logger.debug(f"Failed to extract citation metadata: {e}")

    def _append_table_content(
        self, html_content: str, extracted: dict[str, Any]
    ) -> None:
        """Extract tables and append to extracted content."""
        try:
            from ingestforge.ingest.html_table_extractor import HTMLTableExtractor

            table_extractor = HTMLTableExtractor()
            tables = table_extractor.extract(html_content)

            if not tables:
                return

            table_text_parts = self._format_tables_as_markdown(tables)
            if table_text_parts:
                table_text = "\n".join(table_text_parts)
                extracted["text"] = extracted.get("text", "") + table_text
                extracted["markdown"] = extracted.get("markdown", "") + table_text
        except Exception as e:
            logger.debug(f"Failed to extract tables from HTML: {e}")

    def _format_tables_as_markdown(self, tables: list[Any]) -> list[Any]:
        """Format extracted tables as markdown."""
        table_text_parts = []
        for table in tables:
            md = table.to_markdown()
            if md:
                caption = (
                    f"\n\n### Table: {table.caption}\n" if table.caption else "\n\n"
                )
                table_text_parts.append(caption + md)
        return table_text_parts

    def _build_extracted_html(
        self,
        extracted: dict[str, Any],
        metadata: dict,
        sections: list[Any],
        headings: list,
        source_location: Any,
        source_url: Optional[str],
    ) -> ExtractedHTML:
        """Build final ExtractedHTML result."""
        return ExtractedHTML(
            text=extracted.get("text", ""),
            markdown=extracted.get("markdown", ""),
            html_clean=extracted.get("html_clean", ""),
            title=metadata.get("title", ""),
            authors=metadata.get("authors", []),
            publication_date=metadata.get("date"),
            description=metadata.get("description"),
            site_name=metadata.get("site_name"),
            url=source_url or metadata.get("url"),
            language=metadata.get("language"),
            sections=sections,
            headings=headings,
            source_location=source_location,
            raw_metadata=metadata,
        )

    def process_url(self, url: str) -> ExtractedHTML:
        """
        Fetch and process a URL directly.

        Uses extraction rules if available for site-specific extraction.

        Args:
            url: URL to fetch and process

        Returns:
            ExtractedHTML with content and metadata
        """
        try:
            import trafilatura

            # Fetch the page
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                raise ValueError(f"Could not fetch URL: {url}")

            # Apply extraction rule if available for this URL
            rule_extracted = self._apply_extraction_rule(downloaded, url)

            if rule_extracted:
                extracted, rule_metadata = rule_extracted
                metadata = self._extract_metadata(downloaded, url)
                metadata.update(rule_metadata)
            else:
                extracted = self._extract_content(downloaded, url)
                metadata = self._extract_metadata(downloaded, url)

            # Extract structure
            sections, headings = self._extract_structure(downloaded)

            # Build source location
            source_location = self._build_source_location(
                file_path=None,
                url=url,
                title=metadata.get("title", ""),
                authors=metadata.get("authors", []),
                publication_date=metadata.get("date"),
            )

            return ExtractedHTML(
                text=extracted.get("text", ""),
                markdown=extracted.get("markdown", ""),
                html_clean=extracted.get("html_clean", ""),
                title=metadata.get("title", ""),
                authors=metadata.get("authors", []),
                publication_date=metadata.get("date"),
                description=metadata.get("description"),
                site_name=metadata.get("site_name"),
                url=url,
                language=metadata.get("language"),
                sections=sections,
                headings=headings,
                source_location=source_location,
                raw_metadata=metadata,
            )

        except ImportError:
            raise ImportError(
                "trafilatura is required for URL fetching. "
                "Install with: pip install trafilatura"
            )

    def _read_file(self, file_path: Path) -> str:
        """Read HTML file with encoding detection."""
        # Try common encodings
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue

        # Fallback: read as bytes and decode with errors ignored
        return file_path.read_bytes().decode("utf-8", errors="ignore")

    def _extract_content(
        self,
        html_content: str,
        url: Optional[str] = None,
    ) -> Dict[str, str]:
        """Extract main content using trafilatura."""
        try:
            import trafilatura
            from trafilatura.settings import use_config

            # Configure trafilatura
            config = use_config()
            config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")

            # Extract as plain text
            text = (
                trafilatura.extract(
                    html_content,
                    url=url,
                    include_tables=self.include_tables,
                    include_links=self.include_links,
                    include_images=self.include_images,
                    favor_precision=self.favor_precision,
                    config=config,
                )
                or ""
            )

            # Extract as markdown (with formatting)
            markdown = (
                trafilatura.extract(
                    html_content,
                    url=url,
                    output_format="markdown",
                    include_tables=self.include_tables,
                    include_links=self.include_links,
                    include_images=self.include_images,
                    favor_precision=self.favor_precision,
                    config=config,
                )
                or ""
            )

            # Extract cleaned HTML
            html_clean = (
                trafilatura.extract(
                    html_content,
                    url=url,
                    output_format="html",
                    include_tables=self.include_tables,
                    include_links=self.include_links,
                    favor_precision=self.favor_precision,
                    config=config,
                )
                or ""
            )

            return {
                "text": text,
                "markdown": markdown,
                "html_clean": html_clean,
            }

        except ImportError:
            # Fallback to BeautifulSoup
            return self._extract_content_fallback(html_content)

    def _extract_content_fallback(self, html_content: str) -> Dict[str, str]:
        """Fallback content extraction using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "lxml")

            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Try to find main content
            main = (
                soup.find("article")
                or soup.find("main")
                or soup.find(class_=re.compile(r"article|content|post|entry", re.I))
                or soup.find("body")
            )

            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)

            return {
                "text": text,
                "markdown": text,  # No markdown conversion in fallback
                "html_clean": str(main) if main else "",
            }

        except ImportError:
            # Last resort: regex-based extraction
            text = re.sub(
                r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.I
            )
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.I)
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return {"text": text, "markdown": text, "html_clean": ""}

    def _extract_metadata(
        self,
        html_content: str,
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        try:
            import trafilatura
            from trafilatura import extract_metadata

            metadata = extract_metadata(html_content, default_url=url)

            if metadata:
                return {
                    "title": metadata.title or "",
                    "authors": self._parse_authors(metadata.author),
                    "date": metadata.date,
                    "description": metadata.description,
                    "site_name": metadata.sitename,
                    "url": metadata.url or url,
                    "language": metadata.language,
                    "categories": metadata.categories or [],
                    "tags": metadata.tags or [],
                }

        except (ImportError, Exception) as e:
            logger.debug(f"Failed to extract metadata with trafilatura: {e}")

        # Fallback to BeautifulSoup
        return self._extract_metadata_fallback(html_content, url)

    def _extract_metadata_fallback(
        self,
        html_content: str,
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fallback metadata extraction using BeautifulSoup.

        Rule #4: Reduced from 77 lines to <60 lines via helper extraction
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "lxml")
            return {
                "title": self._extract_title_from_soup(soup),
                "authors": self._extract_authors_from_soup(soup),
                "date": self._extract_date_from_soup(soup),
                "description": self._extract_description_from_soup(soup),
                "site_name": self._extract_site_name_from_soup(soup),
                "url": self._extract_url_from_soup(soup, url),
                "language": soup.html.get("lang") if soup.html else None,
                "categories": [],
                "tags": [],
            }

        except ImportError:
            return self._build_empty_metadata(url)

    def _extract_title_from_soup(self, soup: Any) -> str:
        """
        Extract title from BeautifulSoup object.

        Rule #4: Extracted to reduce function size
        """
        title = ""
        if soup.title:
            title = soup.title.string or ""
        og_title = soup.find("meta", property="og:title")
        if og_title:
            title = og_title.get("content", title)
        return title

    def _extract_authors_from_soup(self, soup: Any) -> List[str]:
        """
        Extract authors from BeautifulSoup object.

        Rule #4: Extracted to reduce function size
        """
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta:
            return self._parse_authors(author_meta.get("content", ""))
        return []

    def _extract_date_from_soup(self, soup: Any) -> Optional[str]:
        """
        Extract date from BeautifulSoup object.

        Rule #4: Extracted to reduce function size
        """
        date_meta = soup.find(
            "meta", attrs={"name": re.compile(r"date|published", re.I)}
        )
        if date_meta:
            return date_meta.get("content")

        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            return time_tag.get("datetime")

        return None

    def _extract_description_from_soup(self, soup: Any) -> Optional[str]:
        """
        Extract description from BeautifulSoup object.

        Rule #4: Extracted to reduce function size
        """
        desc_meta = soup.find("meta", attrs={"name": "description"})
        description = desc_meta.get("content") if desc_meta else None

        og_desc = soup.find("meta", property="og:description")
        if og_desc:
            description = og_desc.get("content", description)

        return description

    def _extract_site_name_from_soup(self, soup: Any) -> Optional[str]:
        """
        Extract site name from BeautifulSoup object.

        Rule #4: Extracted to reduce function size
        """
        og_site = soup.find("meta", property="og:site_name")
        if og_site:
            return og_site.get("content")
        return None

    def _extract_url_from_soup(
        self, soup: Any, fallback_url: Optional[str]
    ) -> Optional[str]:
        """
        Extract canonical URL from BeautifulSoup object.

        Rule #4: Extracted to reduce function size
        """
        canonical = soup.find("link", rel="canonical")
        if canonical:
            return canonical.get("href", fallback_url)
        return fallback_url

    def _build_empty_metadata(self, url: Optional[str]) -> Dict[str, Any]:
        """
        Build empty metadata dict for ImportError fallback.

        Rule #4: Extracted to reduce function size
        """
        return {
            "title": "",
            "authors": [],
            "date": None,
            "description": None,
            "site_name": None,
            "url": url,
            "language": None,
        }

    def _parse_authors(self, author_string: Optional[str]) -> List[str]:
        """Parse author string into list of author names."""
        if not author_string:
            return []

        # Split by common separators
        separators = [",", ";", " and ", " & ", " AND "]
        authors = [author_string]

        for sep in separators:
            new_authors = []
            for author in authors:
                new_authors.extend(author.split(sep))
            authors = new_authors

        # Clean up
        authors = [a.strip() for a in authors if a.strip()]
        return authors

    def _extract_structure(
        self,
        html_content: str,
    ) -> tuple[List[HTMLSection], List[Dict[str, Any]]]:
        """Extract document structure (headings hierarchy).

        Rule #1: Reduced nesting via helper extraction
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "lxml")

            # Find all headings
            headings = self._extract_all_headings(soup)

            # Build section hierarchy
            sections = self._build_section_hierarchy(headings)

            return sections, headings

        except ImportError:
            return [], []

    def _extract_all_headings(self, soup: Any) -> List[Dict[str, Any]]:
        """Extract all heading elements from soup.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        headings: list[str] = []
        for level in range(1, 7):
            self._extract_headings_at_level(soup, level, headings)
        return headings

    def _extract_headings_at_level(
        self, soup: Any, level: int, headings: List[Dict[str, Any]]
    ) -> None:
        """Extract headings at a specific level.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        for heading in soup.find_all(f"h{level}"):
            text = heading.get_text(strip=True)
            if text:
                headings.append(
                    {
                        "level": level,
                        "text": text,
                        "id": heading.get("id"),
                    }
                )

    def _build_section_hierarchy(
        self,
        headings: List[Dict[str, Any]],
    ) -> List[HTMLSection]:
        """Build hierarchical section structure from flat heading list."""
        if not headings:
            return []

        sections = []
        stack: List[HTMLSection] = []

        for h in headings:
            section = HTMLSection(
                level=h["level"],
                title=h["text"],
                content="",  # Would need content extraction per section
                subsections=[],
            )

            # Pop sections from stack that are at same or lower level
            while stack and stack[-1].level >= section.level:
                stack.pop()

            if stack:
                # Add as subsection of parent
                stack[-1].subsections.append(section)
            else:
                # Top-level section
                sections.append(section)

            stack.append(section)

        return sections

    def _build_source_location(
        self,
        file_path: Optional[Path],
        url: Optional[str],
        title: str,
        authors: List[str],
        publication_date: Optional[str],
    ) -> SourceLocation:
        """Build SourceLocation from extracted metadata."""
        author_objs = [Author(name) for name in authors]

        return SourceLocation(
            source_type=SourceType.WEBPAGE,
            title=title,
            authors=author_objs,
            publication_date=publication_date,
            url=url,
            file_path=str(file_path) if file_path else None,
            accessed_date=datetime.now().strftime("%Y-%m-%d"),
        )
