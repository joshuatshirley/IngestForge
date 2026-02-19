#!/usr/bin/env python3
"""HTML metadata parser for citation extraction.

Parses HTML to extract metadata from meta tags, Open Graph, Schema.org,
Dublin Core, and JSON-LD structured data.
"""

import json
from html.parser import HTMLParser
from typing import Callable, Dict, Optional, Any

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class HTMLMetadataParser(HTMLParser):
    """Parse HTML to extract metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.metadata = {}
        self.in_title = False
        self.title_content = []
        self.in_script = False
        self.script_content = []
        self.script_type = None

    def handle_starttag(self, tag: str, attrs: list[Any]) -> None:
        """
        Handle opening HTML tags.

        Rule #1: Dictionary dispatch eliminates nesting
        """
        attrs_dict = dict(attrs)
        tag_handlers = {
            "title": lambda: setattr(self, "in_title", True),
            "meta": lambda: self._handle_meta_tag(attrs_dict),
            "link": lambda: self._handle_link_tag(attrs_dict),
            "script": lambda: self._handle_script_tag(attrs_dict),
        }

        handler = tag_handlers.get(tag)
        if handler:
            handler()

    def _handle_meta_tag(self, attrs_dict: dict[str, Any]) -> None:
        """Handle meta tag extraction."""
        name = attrs_dict.get("name", "").lower()
        prop = attrs_dict.get("property", "").lower()
        content = attrs_dict.get("content", "")

        if name:
            self._process_meta_name(name, content)
        if prop:
            self._process_meta_property(prop, content)

    def _process_meta_name(self, name: str, content: str) -> None:
        """
        Process meta tags by name attribute.

        Rule #1: Dictionary dispatch eliminates nesting
        """
        # Direct field mappings (simple assignment)
        field_mappings = {
            "description": "description",
            "date": "date",
            "dc.date": "date",
            "citation_title": "title",
            "citation_date": "date",
            "citation_publication_date": "date",
            "citation_journal_title": "journal",
            "citation_volume": "volume",
            "citation_issue": "issue",
            "citation_firstpage": "first_page",
            "citation_lastpage": "last_page",
            "citation_doi": "doi",
            "citation_isbn": "isbn",
            "citation_issn": "issn",
            "citation_arxiv_id": "arxiv_id",
            "citation_pmid": "pmid",
            "citation_publisher": "publisher",
            "citation_language": "language",
            "citation_abstract": "abstract",
            "dc.title": "title",
            "dc.publisher": "publisher",
        }

        # Author fields (append to list)
        author_fields = {"author", "citation_author", "dc.creator"}
        if name in field_mappings:
            self.metadata[field_mappings[name]] = content
            return
        if name in author_fields:
            self.metadata.setdefault("authors", []).append(content)
            return
        special_handlers = {
            "keywords": lambda: self._set_keywords(content),
            "dc.identifier": lambda: self._process_dc_identifier(content),
        }

        handler = special_handlers.get(name)
        if handler:
            handler()

    def _set_keywords(self, content: str) -> None:
        """
        Set keywords from meta tag content.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        self.metadata["keywords"] = [k.strip() for k in content.split(",")]

    def _process_dc_identifier(self, content: str) -> None:
        """Process Dublin Core identifier field."""
        if content.startswith("doi:"):
            self.metadata["doi"] = content[4:]
        elif content.startswith("isbn:"):
            self.metadata["isbn"] = content[5:]

    def _set_og_title(self, content: str) -> None:
        """
        Set Open Graph title.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        self.metadata.setdefault("og_title", content)

    def _set_og_description(self, content: str) -> None:
        """
        Set Open Graph description.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        self.metadata.setdefault("og_description", content)

    def _set_og_url(self, content: str) -> None:
        """
        Set Open Graph URL.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        self.metadata["url"] = content

    def _set_site_name(self, content: str) -> None:
        """
        Set site name.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        self.metadata["site_name"] = content

    def _append_article_author(self, content: str) -> None:
        """
        Append article author to list.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        self.metadata.setdefault("authors", []).append(content)

    def _set_article_published_time(self, content: str) -> None:
        """
        Set article published time.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        self.metadata["date"] = content

    def _get_property_handler(self, prop: str) -> Optional[Callable[[str], None]]:
        """
        Get handler function for meta property.

        Rule #1: Dictionary dispatch eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            prop: Property attribute value

        Returns:
            Handler function or None if no handler
        """
        handlers: Dict[str, Callable[[str], None]] = {
            "og:title": self._set_og_title,
            "og:description": self._set_og_description,
            "og:url": self._set_og_url,
            "og:site_name": self._set_site_name,
            "article:author": self._append_article_author,
            "article:published_time": self._set_article_published_time,
        }
        return handlers.get(prop)

    def _process_meta_property(self, prop: str, content: str) -> None:
        """
        Process meta tags by property attribute (Open Graph).

        Rule #1: No nesting - pure dictionary dispatch
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            prop: Property attribute value
            content: Property content value
        """
        handler = self._get_property_handler(prop)
        if handler:
            handler(content)

    def _handle_link_tag(self, attrs_dict: dict[str, Any]) -> None:
        """Handle link tag extraction."""
        rel = attrs_dict.get("rel", "").lower()
        href = attrs_dict.get("href", "")

        if rel == "canonical":
            self.metadata["canonical_url"] = href
        elif rel == "author":
            self.metadata.setdefault("author_links", []).append(href)

    def _handle_script_tag(self, attrs_dict: dict[str, Any]) -> None:
        """Handle script tag for JSON-LD extraction."""
        script_type = attrs_dict.get("type", "")
        if script_type == "application/ld+json":
            self.in_script = True
            self.script_type = "json-ld"
            self.script_content = []

    def handle_endtag(self, tag: str) -> None:
        """
        Handle closing HTML tags.

        Rule #1: Reduced nesting with early returns
        """
        if tag == "title":
            self.in_title = False
            self.metadata["html_title"] = "".join(self.title_content).strip()
            self.title_content = []
            return
        if tag != "script" or not self.in_script:
            return

        self.in_script = False
        if self.script_type == "json-ld":
            self._extract_json_ld()
        self.script_content = []

    def _extract_json_ld(self) -> None:
        """
        Extract and parse JSON-LD data from script content.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        try:
            json_str = "".join(self.script_content)
            ld_data = json.loads(json_str)
            self.metadata["json_ld"] = ld_data
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON-LD metadata: {e}")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_content.append(data)
        elif self.in_script:
            self.script_content.append(data)
