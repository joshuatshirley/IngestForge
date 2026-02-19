"""PDF downloader - Detect and download PDF links from web pages."""

import re
import time
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PDFDownloadResult:
    """Result of downloading a PDF."""

    url: str
    filename: str
    file_path: Optional[Path]
    success: bool
    error: Optional[str] = None
    size_bytes: int = 0


def detect_pdf_links(html_content: str, base_url: str) -> List[str]:
    """
    Detect PDF links in HTML content.

    Args:
        html_content: HTML page content
        base_url: Base URL for resolving relative links

    Returns:
        List of absolute PDF URLs
    """
    pdf_urls = set()

    # Pattern 1: href attributes ending in .pdf
    href_pattern = re.compile(r'href=["\']([^"\']*\.pdf[^"\']*)["\']', re.IGNORECASE)
    for match in href_pattern.finditer(html_content):
        url = match.group(1)
        absolute_url = urljoin(base_url, url)
        pdf_urls.add(absolute_url)

    # Pattern 2: URLs in text ending in .pdf
    url_pattern = re.compile(r'https?://[^\s"\'<>]+\.pdf', re.IGNORECASE)
    for match in url_pattern.finditer(html_content):
        pdf_urls.add(match.group(0))

    return sorted(pdf_urls)


def _validate_pdf_response(
    response: Any, url: str, filename: str, max_size_mb: int
) -> Optional[PDFDownloadResult]:
    """
    Validate PDF response headers.

    Rule #1: Early return for invalid response
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        response: URL response object
        url: PDF URL
        filename: Filename for download
        max_size_mb: Maximum file size in MB

    Returns:
        PDFDownloadResult with error if invalid, None if valid
    """
    # Check content type
    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
        return PDFDownloadResult(
            url=url,
            filename=filename,
            file_path=None,
            success=False,
            error=f"Not a PDF: Content-Type is {content_type}",
        )

    # Check content length
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > max_size_mb * 1024 * 1024:
        return PDFDownloadResult(
            url=url,
            filename=filename,
            file_path=None,
            success=False,
            error=f"File too large: {int(content_length) / (1024*1024):.1f} MB",
        )

    return None


def _download_pdf_chunks(
    response: Any, output_path: Path, url: str, filename: str, max_size_mb: int
) -> PDFDownloadResult:
    """
    Download PDF in chunks with size checking.

    Rule #1: Reduced nesting (max 2 levels)
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        response: URL response object
        output_path: Path to save PDF
        url: PDF URL
        filename: Filename for download
        max_size_mb: Maximum file size in MB

    Returns:
        PDFDownloadResult with download status
    """
    total_bytes = 0

    with open(output_path, "wb") as f:
        while True:
            chunk = response.read(8192)
            if not chunk:
                break

            total_bytes += len(chunk)

            # Check size limit
            if total_bytes > max_size_mb * 1024 * 1024:
                f.close()
                output_path.unlink(missing_ok=True)
                return PDFDownloadResult(
                    url=url,
                    filename=filename,
                    file_path=None,
                    success=False,
                    error=f"Download exceeded {max_size_mb} MB limit",
                )

            f.write(chunk)

    logger.info(f"Downloaded {filename} ({total_bytes} bytes)")

    return PDFDownloadResult(
        url=url,
        filename=filename,
        file_path=output_path,
        success=True,
        size_bytes=total_bytes,
    )


def download_pdf(
    url: str,
    output_dir: Path,
    filename: Optional[str] = None,
    timeout: int = 30,
    max_size_mb: int = 100,
) -> PDFDownloadResult:
    """
    Download a PDF file.

    Rule #4: Reduced from 65 → 38 lines via helper extraction
    """
    import urllib.request
    import urllib.error

    if filename is None:
        filename = _url_to_filename(url)

    output_path = output_dir / filename

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; IngestForge/1.0)",
                "Accept": "application/pdf",
            },
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            validation_error = _validate_pdf_response(
                response, url, filename, max_size_mb
            )
            if validation_error:
                return validation_error

            # Download PDF in chunks
            output_dir.mkdir(parents=True, exist_ok=True)
            return _download_pdf_chunks(
                response, output_path, url, filename, max_size_mb
            )

    except urllib.error.HTTPError as e:
        return _create_http_error_result(url, filename, e)
    except Exception as e:
        return _create_error_result(url, filename, e)


def _create_http_error_result(url: str, filename: str, error: Any) -> PDFDownloadResult:
    """
    Create result for HTTP error.

    Rule #4: Extracted to reduce download_pdf() size
    """
    return PDFDownloadResult(
        url=url,
        filename=filename,
        file_path=None,
        success=False,
        error=f"HTTP {error.code}: {error.reason}",
    )


def _create_error_result(
    url: str, filename: str, error: Exception
) -> PDFDownloadResult:
    """
    Create result for general error.

    Rule #4: Extracted to reduce download_pdf() size
    """
    return PDFDownloadResult(
        url=url,
        filename=filename,
        file_path=None,
        success=False,
        error=str(error),
    )


def download_pdfs_from_page(
    html_content: str,
    base_url: str,
    output_dir: Path,
    max_pdfs: int = 10,
    delay: float = 1.0,
) -> List[PDFDownloadResult]:
    """
    Detect and download all PDF links from an HTML page.

    Args:
        html_content: HTML page content
        base_url: Base URL for resolving relative links
        output_dir: Directory to save PDFs
        max_pdfs: Maximum number of PDFs to download
        delay: Delay between downloads in seconds

    Returns:
        List of PDFDownloadResult
    """
    pdf_urls = detect_pdf_links(html_content, base_url)
    results = []

    for i, url in enumerate(pdf_urls[:max_pdfs]):
        if i > 0:
            time.sleep(delay)

        logger.info(f"Downloading PDF {i+1}/{min(len(pdf_urls), max_pdfs)}: {url}")
        result = download_pdf(url, output_dir)
        results.append(result)

    return results


def _url_to_filename(url: str) -> str:
    """Convert a URL to a safe filename."""
    parsed = urlparse(url)
    path = parsed.path

    # Get the filename from the URL path
    filename = Path(path).name

    if not filename or not filename.endswith(".pdf"):
        # Generate filename from URL
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", path.strip("/"))
        filename = safe[-50:] + ".pdf" if safe else "download.pdf"

    return filename
