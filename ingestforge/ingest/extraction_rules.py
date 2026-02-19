"""
Configurable content extraction rules for different site structures.

Defines extraction rules that can be configured per-site or per-pattern
to handle diverse HTML structures consistently.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class SelectorType(Enum):
    """Type of CSS/XPath selector."""

    CSS = "css"
    XPATH = "xpath"


class ExtractionStrategy(Enum):
    """Strategy for extracting content."""

    TEXT = "text"  # Extract text content
    HTML = "html"  # Extract inner HTML
    ATTRIBUTE = "attribute"  # Extract attribute value
    ALL_TEXT = "all_text"  # Extract all text including children
    MARKDOWN = "markdown"  # Convert to markdown


@dataclass
class Selector:
    """A selector for finding elements."""

    value: str
    type: SelectorType = SelectorType.CSS
    multiple: bool = False  # Whether to select all matching elements

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "type": self.type.value,
            "multiple": self.multiple,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Selector":
        return cls(
            value=data["value"],
            type=SelectorType(data.get("type", "css")),
            multiple=data.get("multiple", False),
        )


@dataclass
class ExtractionField:
    """Definition of a field to extract."""

    name: str
    selector: Selector
    strategy: ExtractionStrategy = ExtractionStrategy.TEXT
    attribute: Optional[str] = None  # For ATTRIBUTE strategy
    default: Optional[str] = None
    required: bool = False
    transform: Optional[str] = None  # Name of transform function
    filters: List[str] = field(default_factory=list)  # Post-processing filters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "selector": self.selector.to_dict(),
            "strategy": self.strategy.value,
            "attribute": self.attribute,
            "default": self.default,
            "required": self.required,
            "transform": self.transform,
            "filters": self.filters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionField":
        return cls(
            name=data["name"],
            selector=Selector.from_dict(data["selector"]),
            strategy=ExtractionStrategy(data.get("strategy", "text")),
            attribute=data.get("attribute"),
            default=data.get("default"),
            required=data.get("required", False),
            transform=data.get("transform"),
            filters=data.get("filters", []),
        )


@dataclass
class ContentBoundary:
    """Defines boundaries for main content area."""

    container: Optional[Selector] = None  # Main content container
    remove: List[Selector] = field(default_factory=list)  # Elements to remove
    keep: List[Selector] = field(default_factory=list)  # Elements to always keep

    def to_dict(self) -> Dict[str, Any]:
        return {
            "container": self.container.to_dict() if self.container else None,
            "remove": [s.to_dict() for s in self.remove],
            "keep": [s.to_dict() for s in self.keep],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentBoundary":
        return cls(
            container=Selector.from_dict(data["container"])
            if data.get("container")
            else None,
            remove=[Selector.from_dict(s) for s in data.get("remove", [])],
            keep=[Selector.from_dict(s) for s in data.get("keep", [])],
        )


@dataclass
class PaginationRule:
    """Rules for handling paginated content."""

    next_page: Optional[Selector] = None  # Next page link
    page_number: Optional[Selector] = None  # Current page number
    total_pages: Optional[Selector] = None  # Total page count
    max_pages: int = 10  # Maximum pages to follow

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_page": self.next_page.to_dict() if self.next_page else None,
            "page_number": self.page_number.to_dict() if self.page_number else None,
            "total_pages": self.total_pages.to_dict() if self.total_pages else None,
            "max_pages": self.max_pages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaginationRule":
        return cls(
            next_page=Selector.from_dict(data["next_page"])
            if data.get("next_page")
            else None,
            page_number=Selector.from_dict(data["page_number"])
            if data.get("page_number")
            else None,
            total_pages=Selector.from_dict(data["total_pages"])
            if data.get("total_pages")
            else None,
            max_pages=data.get("max_pages", 10),
        )


@dataclass
class ExtractionRule:
    """
    Complete extraction rule for a site or page pattern.
    """

    name: str
    description: str = ""
    # URL matching
    url_pattern: Optional[str] = None  # Regex pattern for URL matching
    domain: Optional[str] = None  # Domain restriction
    path_pattern: Optional[str] = None  # Regex for path matching

    # Content boundaries
    boundaries: ContentBoundary = field(default_factory=ContentBoundary)

    # Fields to extract
    fields: List[ExtractionField] = field(default_factory=list)

    # Pagination
    pagination: Optional[PaginationRule] = None

    # Processing options
    wait_for: Optional[Selector] = None  # Wait for element before extraction
    javascript_required: bool = False
    timeout: int = 30  # Seconds

    # Metadata
    version: str = "1.0"
    priority: int = 0  # Higher priority rules match first
    enabled: bool = True

    def matches_url(self, url: str) -> bool:
        """Check if this rule matches the given URL."""
        if not self.enabled:
            return False

        parsed = urlparse(url)

        # Check domain
        if self.domain:
            if not parsed.netloc.endswith(self.domain):
                return False

        # Check URL pattern
        if self.url_pattern:
            if not re.match(self.url_pattern, url):
                return False

        # Check path pattern
        if self.path_pattern:
            if not re.match(self.path_pattern, parsed.path):
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "url_pattern": self.url_pattern,
            "domain": self.domain,
            "path_pattern": self.path_pattern,
            "boundaries": self.boundaries.to_dict(),
            "fields": [f.to_dict() for f in self.fields],
            "pagination": self.pagination.to_dict() if self.pagination else None,
            "wait_for": self.wait_for.to_dict() if self.wait_for else None,
            "javascript_required": self.javascript_required,
            "timeout": self.timeout,
            "version": self.version,
            "priority": self.priority,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionRule":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            url_pattern=data.get("url_pattern"),
            domain=data.get("domain"),
            path_pattern=data.get("path_pattern"),
            boundaries=ContentBoundary.from_dict(data.get("boundaries", {})),
            fields=[ExtractionField.from_dict(f) for f in data.get("fields", [])],
            pagination=PaginationRule.from_dict(data["pagination"])
            if data.get("pagination")
            else None,
            wait_for=Selector.from_dict(data["wait_for"])
            if data.get("wait_for")
            else None,
            javascript_required=data.get("javascript_required", False),
            timeout=data.get("timeout", 30),
            version=data.get("version", "1.0"),
            priority=data.get("priority", 0),
            enabled=data.get("enabled", True),
        )


class ExtractionRuleRegistry:
    """
    Registry for managing extraction rules.

    Loads rules from JSON files and matches them to URLs.
    """

    def __init__(self, rules_dir: Optional[Path] = None) -> None:
        self.rules: List[ExtractionRule] = []
        self._transforms: Dict[str, Callable[[str], str]] = {}
        self._filters: Dict[str, Callable[[str], str]] = {}

        # Register built-in transforms
        self._register_builtins()

        # Load rules from directory if provided
        if rules_dir and rules_dir.exists():
            self.load_from_directory(rules_dir)

    def _register_builtins(self) -> None:
        """Register built-in transforms and filters."""
        # Transforms
        self._transforms["strip"] = str.strip
        self._transforms["lower"] = str.lower
        self._transforms["upper"] = str.upper
        self._transforms["title"] = str.title
        self._transforms["normalize_whitespace"] = lambda s: " ".join(s.split())

        # Filters
        self._filters["strip"] = str.strip
        self._filters["remove_newlines"] = lambda s: s.replace("\n", " ").replace(
            "\r", ""
        )
        self._filters["collapse_whitespace"] = lambda s: re.sub(r"\s+", " ", s)
        self._filters["remove_html_tags"] = lambda s: re.sub(r"<[^>]+>", "", s)
        self._filters["decode_entities"] = self._decode_html_entities
        self._filters["trim_quotes"] = lambda s: s.strip("\"'")

    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities."""
        import html

        return html.unescape(text)

    def add_rule(self, rule: ExtractionRule) -> None:
        """Add a rule to the registry."""
        self.rules.append(rule)
        # Sort by priority (higher first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def find_matching_rule(self, url: str) -> Optional[ExtractionRule]:
        """Find the best matching rule for a URL."""
        for rule in self.rules:
            if rule.matches_url(url):
                return rule
        return None

    def find_all_matching_rules(self, url: str) -> List[ExtractionRule]:
        """Find all matching rules for a URL."""
        return [rule for rule in self.rules if rule.matches_url(url)]

    def load_from_json(self, json_path: Path) -> ExtractionRule:
        """Load a rule from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rule = ExtractionRule.from_dict(data)
        self.add_rule(rule)
        return rule

    def load_from_yaml(self, yaml_path: Path) -> ExtractionRule:
        """
        Load a rule from YAML file.

        Args:
            yaml_path: Path to YAML file containing rule definition.

        Returns:
            ExtractionRule loaded from file.

        Raises:
            ImportError: If PyYAML is not installed.
            ValueError: If YAML file is invalid.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML rules. Install with: pip install pyyaml"
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            # SEC-002: Sanitize path disclosure
            logger.error(f"Invalid YAML rule file: {yaml_path}")
            raise ValueError("Invalid YAML rule file: [REDACTED]")

        rule = ExtractionRule.from_dict(data)
        self.add_rule(rule)
        return rule

    def load_from_directory(self, rules_dir: Path) -> int:
        """
        Load all extraction rules from a directory.

        Scans for .json and .yaml/.yml files and loads each as an extraction rule.

        Args:
            rules_dir: Directory containing rule files.

        Returns:
            Number of rules successfully loaded.
        """
        if not rules_dir.exists() or not rules_dir.is_dir():
            return 0

        loaded_count = 0
        rule_files = self._collect_rule_files(rules_dir)

        for rule_file in rule_files:
            if self._load_rule_file(rule_file):
                loaded_count += 1

        return loaded_count

    def _collect_rule_files(self, rules_dir: Path) -> List[Path]:
        """
        Collect all rule files from directory.

        Args:
            rules_dir: Directory to scan.

        Returns:
            List of paths to rule files (.json, .yaml, .yml).
        """
        rule_files: List[Path] = []
        for pattern in ["*.json", "*.yaml", "*.yml"]:
            rule_files.extend(rules_dir.glob(pattern))
        return rule_files

    def _load_rule_file(self, rule_file: Path) -> bool:
        """
        Load a single rule file.

        Args:
            rule_file: Path to rule file.

        Returns:
            True if rule was loaded successfully, False otherwise.
        """
        try:
            suffix = rule_file.suffix.lower()
            if suffix == ".json":
                self.load_from_json(rule_file)
            elif suffix in (".yaml", ".yml"):
                self.load_from_yaml(rule_file)
            else:
                return False
            return True
        except Exception:
            # Skip invalid rule files - could be logged in future
            return False


# Built-in rules for common sites
BUILTIN_RULES: List[ExtractionRule] = [
    # Wikipedia
    ExtractionRule(
        name="wikipedia",
        description="Wikipedia article extraction",
        domain="wikipedia.org",
        path_pattern=r"^/wiki/(?!Special:|Wikipedia:|Help:|Category:|File:|Template:|Portal:|Draft:)",
        boundaries=ContentBoundary(
            container=Selector("#mw-content-text .mw-parser-output"),
            remove=[
                Selector(".navbox"),
                Selector(".infobox"),
                Selector(".sidebar"),
                Selector(".mw-editsection"),
                Selector(".reference"),
                Selector(".reflist"),
                Selector("#toc"),
                Selector(".hatnote"),
                Selector(".mbox-small"),
            ],
        ),
        fields=[
            ExtractionField(
                name="title",
                selector=Selector("#firstHeading"),
                required=True,
            ),
            ExtractionField(
                name="content",
                selector=Selector("#mw-content-text .mw-parser-output"),
                strategy=ExtractionStrategy.ALL_TEXT,
            ),
            ExtractionField(
                name="categories",
                selector=Selector("#mw-normal-catlinks ul li a", multiple=True),
            ),
            ExtractionField(
                name="last_modified",
                selector=Selector("#footer-info-lastmod"),
                transform="strip",
            ),
        ],
        priority=10,
    ),
    # Medium articles
    ExtractionRule(
        name="medium",
        description="Medium article extraction",
        domain="medium.com",
        boundaries=ContentBoundary(
            container=Selector("article"),
            remove=[
                Selector("[data-testid='headerNav']"),
                Selector("[data-testid='postFooter']"),
                Selector(".pw-multi-vote-count"),
            ],
        ),
        fields=[
            ExtractionField(
                name="title",
                selector=Selector("h1"),
                required=True,
            ),
            ExtractionField(
                name="subtitle",
                selector=Selector("h2[data-testid='storySubtitle']"),
            ),
            ExtractionField(
                name="author",
                selector=Selector("a[data-testid='authorName']"),
            ),
            ExtractionField(
                name="publish_date",
                selector=Selector("span[data-testid='storyPublishDate']"),
            ),
            ExtractionField(
                name="content",
                selector=Selector("article section"),
                strategy=ExtractionStrategy.MARKDOWN,
            ),
            ExtractionField(
                name="tags",
                selector=Selector("a[href*='/tag/']", multiple=True),
            ),
        ],
        priority=10,
    ),
    # GitHub README
    ExtractionRule(
        name="github_readme",
        description="GitHub repository README extraction",
        domain="github.com",
        path_pattern=r"^/[^/]+/[^/]+/?$",
        boundaries=ContentBoundary(
            container=Selector("article.markdown-body"),
        ),
        fields=[
            ExtractionField(
                name="repo_name",
                selector=Selector("[itemprop='name'] a"),
                required=True,
            ),
            ExtractionField(
                name="description",
                selector=Selector("p.f4.my-3"),
            ),
            ExtractionField(
                name="readme",
                selector=Selector("article.markdown-body"),
                strategy=ExtractionStrategy.MARKDOWN,
            ),
            ExtractionField(
                name="stars",
                selector=Selector("#repo-stars-counter-star"),
            ),
            ExtractionField(
                name="forks",
                selector=Selector("#repo-network-counter"),
            ),
            ExtractionField(
                name="topics",
                selector=Selector("a.topic-tag", multiple=True),
            ),
        ],
        priority=10,
    ),
    # arXiv papers
    ExtractionRule(
        name="arxiv_abstract",
        description="arXiv abstract page extraction",
        domain="arxiv.org",
        path_pattern=r"^/abs/",
        fields=[
            ExtractionField(
                name="title",
                selector=Selector("h1.title"),
                required=True,
                filters=["strip"],
            ),
            ExtractionField(
                name="authors",
                selector=Selector("div.authors a", multiple=True),
            ),
            ExtractionField(
                name="abstract",
                selector=Selector("blockquote.abstract"),
                filters=["strip", "collapse_whitespace"],
            ),
            ExtractionField(
                name="arxiv_id",
                selector=Selector("span.arxivid a"),
            ),
            ExtractionField(
                name="subjects",
                selector=Selector("span.primary-subject"),
            ),
            ExtractionField(
                name="submission_date",
                selector=Selector("div.submission-history"),
                transform="strip",
            ),
            ExtractionField(
                name="pdf_link",
                selector=Selector("a.abs-button[href*='/pdf/']"),
                strategy=ExtractionStrategy.ATTRIBUTE,
                attribute="href",
            ),
        ],
        priority=10,
    ),
    # News articles (generic)
    ExtractionRule(
        name="news_article",
        description="Generic news article extraction",
        boundaries=ContentBoundary(
            container=Selector(
                "article, [role='article'], .article-content, .post-content"
            ),
            remove=[
                Selector("nav"),
                Selector("footer"),
                Selector("aside"),
                Selector(".advertisement"),
                Selector(".ad"),
                Selector(".social-share"),
                Selector(".related-articles"),
                Selector(".comments"),
            ],
        ),
        fields=[
            ExtractionField(
                name="title",
                selector=Selector("h1, [itemprop='headline']"),
                required=True,
            ),
            ExtractionField(
                name="author",
                selector=Selector("[rel='author'], .author-name, [itemprop='author']"),
            ),
            ExtractionField(
                name="publish_date",
                selector=Selector("time, [itemprop='datePublished']"),
                strategy=ExtractionStrategy.ATTRIBUTE,
                attribute="datetime",
            ),
            ExtractionField(
                name="content",
                selector=Selector("article, .article-body, .post-content"),
                strategy=ExtractionStrategy.ALL_TEXT,
            ),
        ],
        priority=1,  # Low priority - fallback rule
    ),
    # Documentation sites
    ExtractionRule(
        name="documentation",
        description="Generic documentation page extraction",
        boundaries=ContentBoundary(
            container=Selector("main, .content, .documentation, article"),
            remove=[
                Selector("nav"),
                Selector(".sidebar"),
                Selector(".toc"),
                Selector(".breadcrumb"),
                Selector("footer"),
            ],
        ),
        fields=[
            ExtractionField(
                name="title",
                selector=Selector("h1"),
                required=True,
            ),
            ExtractionField(
                name="content",
                selector=Selector("main, .content, article"),
                strategy=ExtractionStrategy.MARKDOWN,
            ),
            ExtractionField(
                name="headings",
                selector=Selector("h2, h3", multiple=True),
            ),
            ExtractionField(
                name="code_blocks",
                selector=Selector("pre code", multiple=True),
                strategy=ExtractionStrategy.TEXT,
            ),
        ],
        priority=2,
    ),
]


def create_default_registry(rules_dir: Optional[Path] = None) -> ExtractionRuleRegistry:
    """
    Create a registry with built-in rules loaded.

    Args:
        rules_dir: Optional directory with additional custom rules

    Returns:
        ExtractionRuleRegistry with built-in and custom rules
    """
    registry = ExtractionRuleRegistry(rules_dir)

    # Add built-in rules
    for rule in BUILTIN_RULES:
        registry.add_rule(rule)

    return registry


class _RegistrySingleton:
    """Singleton holder for default registry.

    Rule #6: Encapsulates singleton state in smallest scope.
    """

    _instance: Optional[ExtractionRuleRegistry] = None

    @classmethod
    def get(cls) -> ExtractionRuleRegistry:
        """Get or create the default rule registry."""
        if cls._instance is None:
            cls._instance = create_default_registry()
        return cls._instance
