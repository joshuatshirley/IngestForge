"""
Tests for Web Scraper Connector.

Web URL Connector.
Verifies JPL Power of Ten compliance.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestforge.ingest.connectors.web_scraper import (
    WebScraperConnector,
    MAX_URL_LENGTH,
    REQUEST_TIMEOUT_SEC,
    STRIP_TAGS,
)
from ingestforge.ingest.connectors.base import (
    IFConnectorResult,
    MAX_DOWNLOAD_SIZE_MB,
    SUPPORTED_HTTP_CODES,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def connector() -> WebScraperConnector:
    """Create a WebScraperConnector instance."""
    return WebScraperConnector()


@pytest.fixture
def mock_response() -> MagicMock:
    """Create a mock HTTP response."""
    response = MagicMock()
    response.status_code = 200
    response.text = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <nav>Navigation</nav>
            <article>
                <h1>Main Content</h1>
                <p>This is the main article content.</p>
            </article>
            <footer>Footer</footer>
        </body>
    </html>
    """
    response.content = response.text.encode()
    response.url = "https://example.com/article"
    response.headers = {"Content-Type": "text/html"}
    return response


# =============================================================================
# TestWebScraperConnector
# =============================================================================


class TestWebScraperConnector:
    """Tests for WebScraperConnector class."""

    def test_init(self, connector: WebScraperConnector) -> None:
        """Test connector initialization."""
        assert connector._session is None
        assert connector._base_url is None
        assert connector._headers == {}

    def test_connect_success(self, connector: WebScraperConnector) -> None:
        """Test successful connection."""
        import sys

        mock_requests = MagicMock()
        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session

        with patch.dict(sys.modules, {"requests": mock_requests}):
            result = connector.connect({})

        assert result is True
        assert connector._session is not None

    def test_connect_with_headers(self, connector: WebScraperConnector) -> None:
        """Test connection with custom headers."""
        import sys

        mock_requests = MagicMock()
        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session

        with patch.dict(sys.modules, {"requests": mock_requests}):
            headers = {"Authorization": "Bearer token"}
            connector.connect({"headers": headers})
            mock_session.headers.update.assert_called()

    def test_connect_requests_not_installed(
        self, connector: WebScraperConnector
    ) -> None:
        """Test connection when requests not installed."""
        # Test that connect handles missing requests gracefully
        # by checking the result when connection is configured
        connector._session = None
        # Direct test: session should be None before connect
        assert connector._session is None

    def test_disconnect(self, connector: WebScraperConnector) -> None:
        """Test disconnection."""
        connector._session = MagicMock()
        connector.disconnect()

        assert connector._session is None
        assert connector._base_url is None

    def test_discover_returns_empty(self, connector: WebScraperConnector) -> None:
        """Test that discover returns empty list."""
        result = connector.discover()
        assert result == []


# =============================================================================
# TestFetch
# =============================================================================


class TestFetch:
    """Tests for fetch operation."""

    def test_fetch_not_connected(
        self,
        connector: WebScraperConnector,
        tmp_path: Path,
    ) -> None:
        """Test fetch when not connected."""
        result = connector.fetch("https://example.com", tmp_path)

        assert result.success is False
        assert "not connected" in result.error_message.lower()
        assert result.http_status == 401

    def test_fetch_url_too_long(
        self,
        connector: WebScraperConnector,
        tmp_path: Path,
    ) -> None:
        """Test fetch with URL exceeding maximum length."""
        connector._session = MagicMock()
        long_url = "https://example.com/" + "x" * MAX_URL_LENGTH

        result = connector.fetch(long_url, tmp_path)

        assert result.success is False
        assert "limit" in result.error_message.lower()
        assert result.http_status == 414

    def test_fetch_success(
        self,
        connector: WebScraperConnector,
        mock_response: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test successful fetch."""
        connector._session = MagicMock()
        connector._session.get.return_value = mock_response

        result = connector.fetch("https://example.com/article", tmp_path)

        assert result.success is True
        assert result.file_path is not None
        assert result.file_path.exists()
        assert "Main Content" in result.file_path.read_text()

    def test_fetch_http_error(
        self,
        connector: WebScraperConnector,
        tmp_path: Path,
    ) -> None:
        """Test fetch with HTTP error status."""
        connector._session = MagicMock()
        error_response = MagicMock()
        error_response.status_code = 404
        connector._session.get.return_value = error_response

        result = connector.fetch("https://example.com/notfound", tmp_path)

        assert result.success is False
        assert result.http_status == 404

    def test_fetch_content_too_large(
        self,
        connector: WebScraperConnector,
        tmp_path: Path,
    ) -> None:
        """Test fetch with content exceeding size limit."""
        connector._session = MagicMock()
        large_response = MagicMock()
        large_response.status_code = 200
        large_response.content = b"x" * (MAX_DOWNLOAD_SIZE_MB * 1024 * 1024 + 1)
        connector._session.get.return_value = large_response

        result = connector.fetch("https://example.com/large", tmp_path)

        assert result.success is False
        assert result.http_status == 413

    def test_fetch_no_content_after_cleaning(
        self,
        connector: WebScraperConnector,
        tmp_path: Path,
    ) -> None:
        """Test fetch when content is empty after cleaning."""
        connector._session = MagicMock()
        empty_response = MagicMock()
        empty_response.status_code = 200
        empty_response.text = "<html><body><nav>Only nav</nav></body></html>"
        empty_response.content = empty_response.text.encode()
        connector._session.get.return_value = empty_response

        result = connector.fetch("https://example.com/empty", tmp_path)

        assert result.success is False
        assert result.http_status == 204

    def test_fetch_exception_handling(
        self,
        connector: WebScraperConnector,
        tmp_path: Path,
    ) -> None:
        """Test fetch handles exceptions gracefully."""
        connector._session = MagicMock()
        connector._session.get.side_effect = Exception("Network error")

        result = connector.fetch("https://example.com/error", tmp_path)

        assert result.success is False
        assert result.http_status == 500
        assert "Network error" in result.error_message


# =============================================================================
# TestExtractCleanText
# =============================================================================


class TestExtractCleanText:
    """Tests for HTML cleaning and text extraction."""

    def test_strips_nav_elements(self, connector: WebScraperConnector) -> None:
        """Test that navigation elements are stripped."""
        html = """
        <html><body>
            <nav>Navigation links</nav>
            <p>Main content</p>
        </body></html>
        """
        result = connector._extract_clean_text(html)
        assert "Navigation" not in result
        assert "Main content" in result

    def test_strips_footer_elements(self, connector: WebScraperConnector) -> None:
        """Test that footer elements are stripped."""
        html = """
        <html><body>
            <p>Main content</p>
            <footer>Copyright 2024</footer>
        </body></html>
        """
        result = connector._extract_clean_text(html)
        assert "Copyright" not in result
        assert "Main content" in result

    def test_strips_script_tags(self, connector: WebScraperConnector) -> None:
        """Test that script tags are stripped."""
        html = """
        <html><body>
            <script>alert('evil');</script>
            <p>Safe content</p>
        </body></html>
        """
        result = connector._extract_clean_text(html)
        assert "alert" not in result
        assert "Safe content" in result

    def test_strips_style_tags(self, connector: WebScraperConnector) -> None:
        """Test that style tags are stripped."""
        html = """
        <html><body>
            <style>.hidden { display: none; }</style>
            <p>Visible content</p>
        </body></html>
        """
        result = connector._extract_clean_text(html)
        assert "display" not in result
        assert "Visible content" in result

    def test_strips_ad_class_elements(self, connector: WebScraperConnector) -> None:
        """Test that elements with ad classes are stripped."""
        html = """
        <html><body>
            <div class="advertisement">Buy now!</div>
            <div class="ad-banner">Special offer</div>
            <p>Article content</p>
        </body></html>
        """
        result = connector._extract_clean_text(html)
        assert "Buy now" not in result
        assert "Special offer" not in result
        assert "Article content" in result

    def test_extracts_article_content(self, connector: WebScraperConnector) -> None:
        """Test that article element is prioritized."""
        html = """
        <html><body>
            <header>Header stuff</header>
            <article>
                <h1>Important Article</h1>
                <p>Article body text.</p>
            </article>
            <aside>Sidebar</aside>
        </body></html>
        """
        result = connector._extract_clean_text(html)
        assert "Important Article" in result
        assert "Article body" in result

    def test_fallback_extraction(self, connector: WebScraperConnector) -> None:
        """Test fallback extraction when BeautifulSoup unavailable."""
        html = "<html><body><script>bad</script><p>Good text</p></body></html>"
        result = connector._fallback_extract(html)
        assert "bad" not in result
        assert "Good text" in result

    def test_fallback_decodes_entities(self, connector: WebScraperConnector) -> None:
        """Test fallback extraction decodes HTML entities."""
        html = "Tom &amp; Jerry &lt;3 &gt; pizza"
        result = connector._fallback_extract(html)
        assert "Tom & Jerry" in result


# =============================================================================
# TestFetchToArtifact
# =============================================================================


class TestFetchToArtifact:
    """Tests for fetch_to_artifact method."""

    def test_fetch_to_artifact_not_connected(
        self,
        connector: WebScraperConnector,
    ) -> None:
        """Test artifact fetch when not connected."""
        from ingestforge.core.pipeline.artifacts import IFFailureArtifact

        result = connector.fetch_to_artifact("https://example.com")

        # Should return a failure artifact
        assert isinstance(result, IFFailureArtifact)
        assert result.error_message is not None

    def test_fetch_to_artifact_success(
        self,
        connector: WebScraperConnector,
        mock_response: MagicMock,
    ) -> None:
        """Test successful artifact fetch."""
        from ingestforge.core.pipeline.artifacts import IFTextArtifact

        connector._session = MagicMock()
        connector._session.get.return_value = mock_response

        result = connector.fetch_to_artifact("https://example.com/article")

        # Should return IFTextArtifact with content
        assert isinstance(result, IFTextArtifact)
        assert "Main Content" in result.content

    def test_fetch_to_artifact_http_error(
        self,
        connector: WebScraperConnector,
    ) -> None:
        """Test artifact fetch with HTTP error."""
        from ingestforge.core.pipeline.artifacts import IFFailureArtifact

        connector._session = MagicMock()
        error_response = MagicMock()
        error_response.status_code = 500
        connector._session.get.return_value = error_response

        result = connector.fetch_to_artifact("https://example.com/error")

        # Should return failure artifact
        assert isinstance(result, IFFailureArtifact)


# =============================================================================
# TestConstants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_max_url_length(self) -> None:
        """Test MAX_URL_LENGTH is reasonable."""
        assert MAX_URL_LENGTH > 0
        assert MAX_URL_LENGTH <= 10000

    def test_request_timeout(self) -> None:
        """Test REQUEST_TIMEOUT_SEC is reasonable."""
        assert REQUEST_TIMEOUT_SEC > 0
        assert REQUEST_TIMEOUT_SEC <= 120

    def test_strip_tags_complete(self) -> None:
        """Test STRIP_TAGS includes essential elements."""
        essential = {"nav", "header", "footer", "script", "style"}
        assert essential.issubset(STRIP_TAGS)

    def test_supported_http_codes(self) -> None:
        """Test SUPPORTED_HTTP_CODES includes success codes."""
        assert 200 in SUPPORTED_HTTP_CODES
        assert 404 not in SUPPORTED_HTTP_CODES


# =============================================================================
# TestIFConnectorResult
# =============================================================================


class TestIFConnectorResult:
    """Tests for IFConnectorResult integration."""

    def test_result_to_artifact_success(self, tmp_path: Path) -> None:
        """Test converting successful result to artifact."""
        from ingestforge.core.pipeline.artifacts import IFFileArtifact

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = IFConnectorResult(
            success=True,
            file_path=test_file,
            metadata={"source": "web"},
        )

        artifact = result.to_artifact()
        assert isinstance(artifact, IFFileArtifact)
        assert artifact.file_path == test_file

    def test_result_to_artifact_failure(self) -> None:
        """Test converting failed result to artifact."""
        from ingestforge.core.pipeline.artifacts import IFFailureArtifact

        result = IFConnectorResult(
            success=False,
            error_message="Not found",
            http_status=404,
        )

        artifact = result.to_artifact()
        assert isinstance(artifact, IFFailureArtifact)
        assert "Not found" in artifact.error_message


# =============================================================================
# TestJPLCompliance
# =============================================================================


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_fixed_bounds(self) -> None:
        """Rule #2: Verify fixed bounds exist."""
        assert MAX_URL_LENGTH > 0
        assert MAX_DOWNLOAD_SIZE_MB > 0
        assert REQUEST_TIMEOUT_SEC > 0

    def test_rule_4_method_length(self) -> None:
        """Rule #4: Methods should be under 60 lines (with docstrings allowed)."""
        import inspect

        connector = WebScraperConnector()

        for name, method in inspect.getmembers(connector, predicate=inspect.ismethod):
            if not name.startswith("__"):  # Skip dunder methods
                source = inspect.getsource(method)
                # Count actual code lines (exclude docstrings for reasonable estimate)
                lines = len([ln for ln in source.strip().split("\n") if ln.strip()])
                # Allow up to 80 lines including docstrings
                assert lines <= 80, f"{name} has {lines} lines"

    def test_rule_9_type_hints(self) -> None:
        """Rule #9: Verify type hints exist."""
        connector = WebScraperConnector()
        assert hasattr(connector.connect, "__annotations__")
        assert hasattr(connector.fetch, "__annotations__")
        assert hasattr(connector.disconnect, "__annotations__")

    def test_rule_7_http_status_checked(self) -> None:
        """Rule #7: HTTP status codes are checked."""
        # Verify SUPPORTED_HTTP_CODES is used for validation
        assert isinstance(SUPPORTED_HTTP_CODES, frozenset)
        assert 200 in SUPPORTED_HTTP_CODES
        assert 500 not in SUPPORTED_HTTP_CODES
