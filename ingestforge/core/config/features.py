"""
Feature-specific configuration.

Provides configuration for specialized features: API server, OCR processing,
web search, research sessions, literary analysis, and feature analysis with
Army doctrine integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class APIConfig:
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key: str = ""

    # mTLS & SSL settings ()
    mtls_enabled: bool = False
    ssl_certfile: str = ""
    ssl_keyfile: str = ""
    ssl_ca_certs: str = ""  # For validating clients
    trust_store_path: str = "trust_store.yaml"


@dataclass
class LiteraryScrapingConfig:
    """Scraping settings for literary reference gathering."""

    delay_min: float = 1.0
    delay_max: float = 3.0
    respect_robots: bool = True
    max_pages_per_wiki: int = 50


@dataclass
class LiteraryAnalysisConfig:
    """Analysis settings for literary commands."""

    use_llm: bool = False
    min_evidence_threshold: float = 0.3


@dataclass
class LiteraryConfig:
    """Configuration for literary analysis features."""

    default_citation_style: str = "mla"
    scraping: LiteraryScrapingConfig = field(default_factory=LiteraryScrapingConfig)
    analysis: LiteraryAnalysisConfig = field(default_factory=LiteraryAnalysisConfig)


@dataclass
class OCRConfig:
    """OCR processing configuration."""

    preferred_engine: str = "auto"  # "auto", "tesseract", "easyocr"
    language: str = "eng"  # Tesseract lang code
    languages: List[str] = field(default_factory=lambda: ["en"])  # EasyOCR codes
    scanned_threshold: int = 100  # chars/page for scanned detection
    confidence_threshold: float = 0.3  # min confidence to keep result
    use_gpu: bool = False  # GPU for EasyOCR
    page_timeout: int = 120  # seconds per page OCR timeout (0=no limit)
    max_workers: int = 1  # concurrent OCR workers for multi-page PDFs
    ocr_embedded_images: bool = True  # OCR images embedded in text-rich pages

    def __post_init__(self) -> None:
        """Validate OCR configuration."""
        valid_engines = {"auto", "tesseract", "easyocr"}
        if self.preferred_engine.lower() not in valid_engines:
            raise ValueError(
                f"Invalid preferred_engine: '{self.preferred_engine}'. "
                f"Valid options: {', '.join(sorted(valid_engines))}"
            )
        if self.scanned_threshold < 0:
            raise ValueError("scanned_threshold must be non-negative")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.page_timeout < 0:
            raise ValueError("page_timeout must be non-negative")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")


@dataclass
class WebSearchConfig:
    """Web search configuration for research sessions."""

    max_results: int = 20
    region: str = "us-en"
    safe_search: str = "moderate"
    educational_boost: bool = True
    educational_domains: List[str] = field(
        default_factory=lambda: [
            ".edu",
            ".gov",
            "wikipedia.org",
            "arxiv.org",
            "scholar.google.com",
            "pubmed.ncbi.nlm.nih.gov",
            "jstor.org",
            "ncbi.nlm.nih.gov",
        ]
    )
    excluded_domains: List[str] = field(
        default_factory=lambda: [
            "pinterest.com",
            "facebook.com",
            "twitter.com",
            "x.com",
            "instagram.com",
            "tiktok.com",
            "reddit.com",
            "amazon.com",
            "ebay.com",
            "etsy.com",
        ]
    )


@dataclass
class ResearchConfig:
    """Research session configuration."""

    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    auto_process: bool = True
    max_sources_per_session: int = 50
    default_scrape_delay_min: float = 1.0
    default_scrape_delay_max: float = 3.0


@dataclass
class DoctrineAPIConfig:
    """Army Doctrine RAG API configuration.

    Used by the feature analyzer to query regulations and policies
    that may apply to feature implementations.
    """

    url: str = "http://localhost:8000"
    enabled: bool = True
    timeout_seconds: int = 30
    top_k: int = 5  # Default number of results to retrieve


@dataclass
class FeatureAnalysisConfig:
    """Feature analysis configuration."""

    doctrine_api: DoctrineAPIConfig = field(default_factory=DoctrineAPIConfig)
    max_code_results: int = 20
    max_story_results: int = 10
    max_generated_stories: int = 5


@dataclass
class RedactionConfig:
    """PII redaction configuration (SEC-001.2).

    Defines which PII types to redact and terms to skip.

    Example config.yaml:
        redaction:
          enabled: true
          types:
            - email
            - phone
            - ssn
            - person_name
          whitelist:
            - support@company.com
            - John Smith
          mask_char: "*"
          preserve_length: false
          show_type: true
    """

    enabled: bool = False
    types: List[str] = field(
        default_factory=lambda: ["email", "phone", "ssn", "person_name"]
    )
    whitelist: List[str] = field(default_factory=list)
    mask_char: str = "*"
    preserve_length: bool = False
    show_type: bool = True
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    max_whitelist_entries: int = 1000
    max_custom_patterns: int = 10

    def __post_init__(self) -> None:
        """Validate redaction configuration."""
        valid_types = {
            "email",
            "phone",
            "ssn",
            "credit_card",
            "person_name",
            "address",
            "date_of_birth",
            "ip_address",
            "custom",
        }
        for pii_type in self.types:
            if pii_type.lower() not in valid_types:
                raise ValueError(
                    f"Invalid PII type: '{pii_type}'. "
                    f"Valid types: {', '.join(sorted(valid_types))}"
                )
        if len(self.whitelist) > self.max_whitelist_entries:
            raise ValueError(
                f"Whitelist exceeds max entries ({self.max_whitelist_entries})"
            )
        if len(self.custom_patterns) > self.max_custom_patterns:
            raise ValueError(
                f"Custom patterns exceeds max ({self.max_custom_patterns})"
            )
