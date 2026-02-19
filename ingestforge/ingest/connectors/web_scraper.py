"""Web URL Scraper IConnector Implementation.

Fetches web pages, strips ads/navigation, and produces clean IFTextArtifact.
Follows NASA JPL Rules #4 (Modular), #7 (Check Returns), #9 (Type Hints).

Web scraper strips ads/nav and produces clean IFTextArtifact.
"""

from __future__ import annotations
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from ingestforge.core.logging import get_logger
from ingestforge.ingest.connectors.base import (
    IConnector,
    IFConnectorResult,
    MAX_DOWNLOAD_SIZE_MB,
    SUPPORTED_HTTP_CODES,
)

logger = get_logger(__name__)

MAX_URL_LENGTH = 2048
REQUEST_TIMEOUT_SEC = 30
MAX_REDIRECTS = 5

# Elements to strip from HTML
STRIP_TAGS = frozenset(
    {
        "nav",
        "header",
        "footer",
        "aside",
        "script",
        "style",
        "noscript",
        "iframe",
        "form",
        "button",
        "input",
        "select",
        "textarea",
        "advertisement",
        "ad",
        "ads",
        "banner",
        "sidebar",
        "menu",
    }
)

# Class patterns indicating ads/nav
AD_CLASS_PATTERNS = frozenset(
    {
        r"ad[-_]?",
        r"advertisement",
        r"banner",
        r"sidebar",
        r"nav[-_]?",
        r"menu",
        r"footer",
        r"header",
        r"social[-_]?",
        r"share[-_]?",
        r"cookie[-_]?",
        r"popup",
        r"modal",
        r"overlay",
    }
)


class WebScraperConnector(IConnector):
    """
    Web URL scraper implementing IF-Protocol.

    Rule #7: Check HTTP return codes; return IFFailureArtifact on 404/500.
    Rule #2: MAX_URL_LENGTH bounds input.
    """

    def __init__(self) -> None:
        """Initialize scraper state."""
        self._session: Any = None
        self._base_url: Optional[str] = None
        self._headers: Dict[str, str] = {}

    def connect(self, config: Dict[str, Any]) -> bool:
        """
        Initialize HTTP session.

        Rule #7: Validate config and establish session.
        """
        try:
            import requests

            self._session = requests.Session()
            self._headers = config.get(
                "headers",
                {
                    "User-Agent": "IngestForge/1.0 (Research Bot)",
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            self._session.headers.update(self._headers)
            self._base_url = config.get("base_url")
            return True
        except ImportError:
            logger.error("requests library not installed")
            return False
        except Exception as e:
            logger.error(f"WebScraper connect failed: {e}")
            return False

    def discover(self) -> List[Dict[str, Any]]:
        """
        Web scraper does not support discovery.

        URLs must be provided explicitly via fetch().
        """
        return []

    def fetch(self, document_id: str, output_dir: Path) -> IFConnectorResult:
        """
        Fetch URL, clean content, and save to output directory.

        Rule #7: Check HTTP status code and return failure on error.
        """
        url = document_id
        if not self._session:
            return IFConnectorResult(
                success=False,
                error_message="Connector not connected",
                http_status=401,
            )

        # Validate URL length
        if len(url) > MAX_URL_LENGTH:
            return IFConnectorResult(
                success=False,
                error_message=f"URL exceeds {MAX_URL_LENGTH} character limit",
                http_status=414,
            )

        try:
            response = self._session.get(
                url,
                timeout=REQUEST_TIMEOUT_SEC,
                allow_redirects=True,
            )

            # Rule #7: Check HTTP status code
            if response.status_code not in SUPPORTED_HTTP_CODES:
                return IFConnectorResult(
                    success=False,
                    error_message=f"HTTP {response.status_code}",
                    http_status=response.status_code,
                )

            # Check content size
            content_length = len(response.content)
            if content_length > MAX_DOWNLOAD_SIZE_MB * 1024 * 1024:
                return IFConnectorResult(
                    success=False,
                    error_message=f"Content exceeds {MAX_DOWNLOAD_SIZE_MB}MB",
                    http_status=413,
                )

            # Clean and extract text
            clean_text = self._extract_clean_text(response.text)
            if not clean_text.strip():
                return IFConnectorResult(
                    success=False,
                    error_message="No content extracted after cleaning",
                    http_status=204,
                )

            # Generate filename from URL
            parsed = urlparse(url)
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            file_name = f"{parsed.netloc}_{url_hash}.txt"

            # Write to output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / file_name
            output_file.write_text(clean_text, encoding="utf-8")

            return IFConnectorResult(
                success=True,
                file_path=output_file,
                http_status=response.status_code,
                metadata={
                    "source": "web",
                    "url": url,
                    "content_type": response.headers.get("Content-Type"),
                    "final_url": response.url,
                },
            )
        except Exception as e:
            logger.error(f"WebScraper fetch failed: {e}")
            return IFConnectorResult(
                success=False,
                error_message=str(e),
                http_status=500,
            )

    def _extract_clean_text(self, html: str) -> str:
        """
        Strip ads, navigation, and extract clean text.

        Rule #4: Isolated cleaning logic.
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove unwanted tags
            for tag in STRIP_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()

            # Remove elements with ad-related classes/IDs
            for pattern in AD_CLASS_PATTERNS:
                for element in soup.find_all(class_=re.compile(pattern, re.IGNORECASE)):
                    element.decompose()
                for element in soup.find_all(id=re.compile(pattern, re.IGNORECASE)):
                    element.decompose()

            # Extract text from article, main, or body
            content = soup.find("article") or soup.find("main") or soup.body
            if content:
                text = content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n")]
            lines = [line for line in lines if line]
            return "\n".join(lines)
        except ImportError:
            logger.warning("beautifulsoup4 not installed, using fallback")
            return self._fallback_extract(html)
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return self._fallback_extract(html)

    def _fallback_extract(self, html: str) -> str:
        """
        Fallback text extraction without BeautifulSoup.

        Rule #4: Simple regex-based extraction.
        """
        # Remove script and style content
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Decode common entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fetch_to_artifact(self, url: str) -> "IFTextArtifact":
        """
        Fetch URL directly to IFTextArtifact without file I/O.

        Produces clean IFTextArtifact.
        """
        from ingestforge.core.pipeline.artifacts import (
            IFTextArtifact,
            IFFailureArtifact,
        )

        if not self._session:
            return IFFailureArtifact(
                error_message="Connector not connected",
                metadata={"http_status": 401},
            )

        try:
            response = self._session.get(url, timeout=REQUEST_TIMEOUT_SEC)

            if response.status_code not in SUPPORTED_HTTP_CODES:
                return IFFailureArtifact(
                    error_message=f"HTTP {response.status_code}",
                    metadata={"http_status": response.status_code, "url": url},
                )

            clean_text = self._extract_clean_text(response.text)
            return IFTextArtifact(
                content=clean_text,
                metadata={
                    "source": "web",
                    "url": url,
                    "final_url": response.url,
                },
            )
        except Exception as e:
            return IFFailureArtifact(
                error_message=str(e),
                metadata={"url": url},
            )

    def disconnect(self) -> None:
        """Clean up session."""
        if self._session:
            self._session.close()
        self._session = None
        self._base_url = None
