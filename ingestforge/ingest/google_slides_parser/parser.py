"""
Google Slides parser for presentations.

Provides GoogleSlidesParser for parsing Google Slides API responses.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from ingestforge.core.logging import get_logger
from ingestforge.ingest.google_slides_parser.extractors import (
    extract_text_from_shape,
    parse_element,
)
from ingestforge.ingest.google_slides_parser.models import ParsedSlide, SlideElementType
from ingestforge.ingest.google_slides_parser.presentation import ParsedPresentation

logger = get_logger(__name__)


class GoogleSlidesParser:
    """
    Parse Google Slides presentations.

    Supports parsing from API response JSON or directly from
    presentation ID using the Google Slides API.
    """

    def __init__(self, credentials_path: Optional[Path] = None) -> None:
        """
        Initialize the parser.

        Args:
            credentials_path: Path to OAuth2 credentials JSON file
        """
        self.credentials_path = credentials_path
        self._service = None
        self._credentials = None

    def parse_from_json(self, api_response: Dict[str, Any]) -> ParsedPresentation:
        """
        Parse a presentation from API response JSON.

        Args:
            api_response: JSON response from Google Slides API

        Returns:
            ParsedPresentation object
        """
        presentation = ParsedPresentation(
            presentation_id=api_response.get("presentationId", ""),
            title=api_response.get("title", "Untitled"),
            locale=api_response.get("locale", "en"),
            fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        # Page size
        page_size = api_response.get("pageSize", {})
        presentation.page_width = self._emu_to_points(
            page_size.get("width", {}).get("magnitude", 0)
        )
        presentation.page_height = self._emu_to_points(
            page_size.get("height", {}).get("magnitude", 0)
        )

        # Parse slides
        slides_data = api_response.get("slides", [])
        all_text = []
        all_notes = []
        all_images = []

        for idx, slide_data in enumerate(slides_data):
            slide = self._parse_slide(slide_data, idx)
            presentation.slides.append(slide)

            # Collect text and notes
            if slide.title:
                all_text.append(slide.title)
            if slide.body_text:
                all_text.append(slide.body_text)
            if slide.speaker_notes:
                all_notes.append(slide.speaker_notes)

            # Collect image URLs
            for element in slide.elements:
                if element.image_url:
                    all_images.append(element.image_url)

        presentation.all_text = "\n\n".join(all_text)
        presentation.all_speaker_notes = "\n\n".join(all_notes)
        presentation.image_urls = all_images

        return presentation

    def _parse_slide(self, slide_data: Dict[str, Any], index: int) -> ParsedSlide:
        """
        Parse a single slide.

        Rule #1: Reduced nesting with helper methods
        """
        slide = ParsedSlide(
            slide_id=slide_data.get("objectId", ""),
            index=index,
        )

        # Layout
        slide_props = slide_data.get("slideProperties", {})
        layout_ref = slide_props.get("layoutObjectId")
        if layout_ref:
            slide.layout_id = layout_ref
        text_parts = self._parse_slide_elements(slide, slide_data)
        self._extract_slide_title(slide, text_parts)
        self._extract_slide_body(slide)
        self._extract_speaker_notes(slide, slide_props)

        return slide

    def _parse_slide_elements(
        self, slide: ParsedSlide, slide_data: Dict[str, Any]
    ) -> List[str]:
        """
        Parse all page elements in a slide.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        elements = slide_data.get("pageElements", [])
        text_parts = []

        for elem_data in elements:
            element = parse_element(elem_data)
            if element:
                slide.elements.append(element)

                # Collect text
                if element.element_type == SlideElementType.TEXT and element.plain_text:
                    text_parts.append(element.plain_text)

        return text_parts

    def _extract_slide_title(self, slide: ParsedSlide, text_parts: List[str]) -> None:
        """
        Extract slide title from elements.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        # Try to find title element first
        for element in slide.elements:
            if element.element_type != SlideElementType.TEXT:
                continue

            obj_id = element.object_id.lower()
            if "title" in obj_id or "header" in obj_id:
                slide.title = element.plain_text.strip()
                return
        if not text_parts:
            return

        # Use first significant text as title
        for part in text_parts:
            if part.strip():
                lines = part.strip().split("\n")
                slide.title = lines[0] if lines else None
                return

    def _extract_slide_body(self, slide: ParsedSlide) -> None:
        """
        Extract slide body text.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        body_parts = []
        for element in slide.elements:
            if element.element_type != SlideElementType.TEXT:
                continue

            text = element.plain_text.strip()
            if text and text != slide.title:
                body_parts.append(text)

        slide.body_text = "\n".join(body_parts)

    def _extract_speaker_notes(
        self, slide: ParsedSlide, slide_props: Dict[str, Any]
    ) -> None:
        """
        Extract speaker notes from slide.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        notes_page = slide_props.get("notesPage", {})
        notes_elements = notes_page.get("pageElements", [])

        for notes_elem in notes_elements:
            shape = notes_elem.get("shape", {})
            shape_type = shape.get("shapeType")
            if shape_type != "TEXT_BOX":
                continue

            text = extract_text_from_shape(shape)
            # Skip auto-generated slide numbers
            if text and not text.startswith("Slide"):
                slide.speaker_notes = text
                return

    def _emu_to_points(self, emu: float) -> float:
        """Convert EMU (English Metric Units) to points."""
        # 1 point = 12700 EMU
        return emu / 12700.0 if emu else 0.0

    @staticmethod
    def extract_presentation_id(url_or_id: str) -> Optional[str]:
        """
        Extract presentation ID from URL or return ID if already valid.

        Rule #1: Reduced nesting with helper methods

        Args:
            url_or_id: Google Slides URL or presentation ID

        Returns:
            Presentation ID or None if invalid
        """
        if re.match(r"^[\w-]{20,}$", url_or_id):
            return url_or_id
        return GoogleSlidesParser._extract_id_from_url(url_or_id)

    @staticmethod
    def _extract_id_from_url(url: str) -> Optional[str]:
        """
        Extract presentation ID from Google URL.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        try:
            parsed = urlparse(url)
            if "google.com" in parsed.netloc:
                pres_id = GoogleSlidesParser._extract_from_google_docs(parsed.path)
                if pres_id:
                    return pres_id
            if "drive.google.com" in parsed.netloc:
                pres_id = GoogleSlidesParser._extract_from_google_drive(parsed)
                if pres_id:
                    return pres_id

        except Exception as e:
            logger.debug(f"Failed to extract presentation ID from URL: {e}")

        return None

    @staticmethod
    def _extract_from_google_docs(path: str) -> Optional[str]:
        """
        Extract ID from docs.google.com/presentation URLs.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        path_parts = path.split("/")
        for i, part in enumerate(path_parts):
            if part == "d" and i + 1 < len(path_parts):
                return path_parts[i + 1]
        return None

    @staticmethod
    def _extract_from_google_drive(parsed: Any) -> Optional[str]:
        """
        Extract ID from drive.google.com URLs.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        # Handle /file/d/ format
        if "/file/d/" in parsed.path:
            match = re.search(r"/d/([\w-]+)", parsed.path)
            if match:
                return match.group(1)

        # Handle open?id= format
        query = parse_qs(parsed.query)
        if "id" in query:
            return query["id"][0]

        return None

    def fetch_presentation(
        self,
        presentation_id: str,
    ) -> Optional[ParsedPresentation]:
        """
        Fetch and parse a presentation using Google Slides API.

        Requires valid OAuth2 credentials.

        Args:
            presentation_id: Presentation ID

        Returns:
            ParsedPresentation or None if fetch failed
        """
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Google API client libraries required. Install with: "
                "pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
            )

        if not self.credentials_path:
            raise ValueError("Credentials path required for API access")

        # Load or refresh credentials
        creds = self._get_credentials()
        if not creds:
            raise RuntimeError("Failed to obtain credentials")

        # Build service
        service = build("slides", "v1", credentials=creds)

        # Fetch presentation
        response = service.presentations().get(presentationId=presentation_id).execute()

        return self.parse_from_json(response)

    def _get_credentials(self) -> None:
        """Get or refresh OAuth2 credentials."""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
        except ImportError:
            return None

        SCOPES = ["https://www.googleapis.com/auth/presentations.readonly"]
        creds = None
        token_path = self.credentials_path.parent / "token.json"

        # Load existing token
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save token
            with open(token_path, "w") as f:
                f.write(creds.to_json())

        return creds
