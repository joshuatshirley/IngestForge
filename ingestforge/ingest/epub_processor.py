"""EPUB document processor.

Processes EPUB (Electronic Publication) e-book files.
Supports EPUB 2.0 and 3.0 formats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import zipfile
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class EPUBProcessor:
    """Process EPUB e-book files."""

    def __init__(self) -> None:
        """Initialize EPUB processor."""
        self.namespaces = {
            "opf": "http://www.idpf.org/2007/opf",
            "dc": "http://purl.org/dc/elements/1.1/",
            "xhtml": "http://www.w3.org/1999/xhtml",
        }

    def process(self, file_path: Path) -> Dict[str, Any]:
        """Process EPUB file.

        Args:
            file_path: Path to EPUB file

        Returns:
            Dictionary with extracted content and metadata
        """
        if not zipfile.is_zipfile(file_path):
            # SEC-002: Sanitize path disclosure
            logger.error(f"Not a valid EPUB file: {file_path}")
            raise ValueError("Not a valid EPUB file: [REDACTED]")

        with zipfile.ZipFile(file_path, "r") as epub:
            # Get content.opf path
            container_path = "META-INF/container.xml"
            opf_path = self._get_opf_path(epub, container_path)

            # Extract metadata
            metadata = self._extract_metadata(epub, opf_path)

            # Extract text content
            text_content = self._extract_text(epub, opf_path)

            # Extract table of contents
            toc = self._extract_toc(epub, opf_path)

            return {
                "text": text_content,
                "metadata": metadata,
                "toc": toc,
                "type": "epub",
                "source": str(file_path.name),
            }

    def _get_opf_path(self, epub: zipfile.ZipFile, container_path: str) -> str:
        """Get path to content.opf file.

        Args:
            epub: ZipFile object
            container_path: Path to container.xml

        Returns:
            Path to content.opf within EPUB
        """
        try:
            container_xml = epub.read(container_path)
            root = ET.fromstring(container_xml)

            # Find rootfile element
            rootfile = root.find(
                ".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile"
            )

            if rootfile is not None:
                return rootfile.get("full-path", "content.opf")

        except Exception as e:
            logger.warning(f"Failed to parse container.xml: {e}. Using default path.")

        # Default fallback
        return "content.opf"

    def _extract_metadata(self, epub: zipfile.ZipFile, opf_path: str) -> Dict[str, Any]:
        """Extract metadata from EPUB.

        Args:
            epub: ZipFile object
            opf_path: Path to content.opf

        Returns:
            Metadata dictionary
        """
        try:
            opf_content = epub.read(opf_path)
            root = ET.fromstring(opf_content)

            metadata = {
                "title": self._get_metadata_field(root, "dc:title"),
                "author": self._get_metadata_field(root, "dc:creator"),
                "publisher": self._get_metadata_field(root, "dc:publisher"),
                "date": self._get_metadata_field(root, "dc:date"),
                "language": self._get_metadata_field(root, "dc:language"),
                "isbn": self._get_metadata_field(root, "dc:identifier"),
            }

            return {k: v for k, v in metadata.items() if v}

        except Exception:
            return {}

    def _get_metadata_field(self, root: ET.Element, field: str) -> Optional[str]:
        """Get metadata field value.

        Args:
            root: XML root element
            field: Field name to extract

        Returns:
            Field value or None
        """
        elem = root.find(f".//opf:metadata/{field}", self.namespaces)
        return elem.text if elem is not None else None

    def _extract_text(self, epub: zipfile.ZipFile, opf_path: str) -> str:
        """Extract text content from EPUB.

        Args:
            epub: ZipFile object
            opf_path: Path to content.opf

        Returns:
            Extracted text content
        """
        try:
            # Get list of content files from manifest
            content_files = self._get_content_files(epub, opf_path)

            # Extract text from each file
            text_parts = []
            for file_path in content_files[:100]:  # Limit to 100 files
                text = self._extract_text_from_file(epub, file_path)
                if text:
                    text_parts.append(text)

            return "\n\n".join(text_parts)

        except Exception:
            return ""

    def _get_content_files(self, epub: zipfile.ZipFile, opf_path: str) -> List[str]:
        """Get ordered list of content files.

        Args:
            epub: ZipFile object
            opf_path: Path to content.opf

        Returns:
            List of content file paths
        """
        try:
            opf_content = epub.read(opf_path)
            root = ET.fromstring(opf_content)

            # Get base directory
            base_dir = str(Path(opf_path).parent)

            # Get spine order
            spine = root.find(".//opf:spine", self.namespaces)
            if spine is None:
                return []

            content_files = []
            for itemref in spine.findall(".//opf:itemref", self.namespaces):
                file_path = self._process_itemref(itemref, root, base_dir)
                if file_path:
                    content_files.append(file_path)

            return content_files

        except Exception:
            return []

    def _process_itemref(self, itemref: Any, root: Any, base_dir: str) -> Optional[str]:
        """Process single itemref to get file path.

        Args:
            itemref: Itemref element
            root: OPF root element
            base_dir: Base directory path

        Returns:
            File path or None
        """
        idref = itemref.get("idref")
        if not idref:
            return None

        # Find corresponding item in manifest
        item = root.find(f'.//opf:manifest/opf:item[@id="{idref}"]', self.namespaces)
        if item is None:
            return None

        href = item.get("href", "")
        if not href:
            return None

        return f"{base_dir}/{href}" if base_dir != "." else href

    def _extract_text_from_file(self, epub: zipfile.ZipFile, file_path: str) -> str:
        """Extract text from single content file.

        Args:
            epub: ZipFile object
            file_path: Path to content file

        Returns:
            Extracted text
        """
        try:
            content = epub.read(file_path).decode("utf-8", errors="ignore")

            # Parse HTML/XHTML
            # Remove tags and extract text (simple approach)
            text = self._strip_html_tags(content)

            return text.strip()

        except Exception:
            return ""

    def _strip_html_tags(self, html: str) -> str:
        """Strip HTML tags from content.

        Args:
            html: HTML content

        Returns:
            Plain text
        """
        # Simple tag stripping (for production, use BeautifulSoup)
        import re

        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _extract_toc(
        self, epub: zipfile.ZipFile, opf_path: str
    ) -> List[Dict[str, str]]:
        """Extract table of contents.

        Args:
            epub: ZipFile object
            opf_path: Path to content.opf

        Returns:
            List of TOC entries
        """
        # Try to find NCX file
        try:
            opf_content = epub.read(opf_path)
            root = ET.fromstring(opf_content)

            # Find NCX reference
            ncx_item = root.find(
                './/opf:manifest/opf:item[@media-type="application/x-dtbncx+xml"]',
                self.namespaces,
            )

            if ncx_item is not None:
                ncx_path = ncx_item.get("href", "")
                if ncx_path:
                    base_dir = str(Path(opf_path).parent)
                    full_ncx_path = (
                        f"{base_dir}/{ncx_path}" if base_dir != "." else ncx_path
                    )
                    return self._parse_ncx(epub, full_ncx_path)

        except Exception as e:
            logger.warning(f"Failed to extract table of contents: {e}")

        return []

    def _parse_ncx(self, epub: zipfile.ZipFile, ncx_path: str) -> List[Dict[str, str]]:
        """Parse NCX navigation file.

        Args:
            epub: ZipFile object
            ncx_path: Path to NCX file

        Returns:
            List of navigation entries
        """
        try:
            ncx_content = epub.read(ncx_path)
            root = ET.fromstring(ncx_content)

            toc = []
            for navpoint in root.findall(
                ".//{http://www.daisy.org/z3986/2005/ncx/}navPoint"
            ):
                label_elem = navpoint.find(
                    ".//{http://www.daisy.org/z3986/2005/ncx/}text"
                )
                content_elem = navpoint.find(
                    ".//{http://www.daisy.org/z3986/2005/ncx/}content"
                )

                if label_elem is not None and content_elem is not None:
                    toc.append(
                        {
                            "label": label_elem.text or "",
                            "href": content_elem.get("src", ""),
                        }
                    )

            return toc

        except Exception:
            return []


def extract_text(file_path: Path) -> str:
    """Extract text from EPUB file.

    Args:
        file_path: Path to EPUB file

    Returns:
        Extracted text content
    """
    processor = EPUBProcessor()
    result = processor.process(file_path)
    text: str = result.get("text", "")
    return text


def extract_with_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract text and metadata from EPUB file.

    Args:
        file_path: Path to EPUB file

    Returns:
        Dictionary with text and metadata
    """
    processor = EPUBProcessor()
    return processor.process(file_path)
